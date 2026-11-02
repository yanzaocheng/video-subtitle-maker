from pathlib import Path
import base64
import json
import os
import queue
import subprocess
import threading
import time

from common import Cancelled, UserError, check_cancel

ROOT = Path(__file__).resolve().parent
GPU_PYTHON = ROOT / "gpu-env" / "Scripts" / "python.exe"
WORKER = ROOT / "local_asr_worker.py"
MODELS = ROOT / "models"
_gpu_cache = {"at": 0.0, "value": None}


def local_status(folder, python=GPU_PYTHON, models=MODELS):
    if not Path(python).is_file():
        return False, "本地 GPU 环境未安装"
    model = Path(models) / folder
    if not (model / "verified.json").is_file() or not (model / "model.safetensors").is_file():
        return False, "本地模型未下载"
    return True, ""


def gpu_memory():
    """Free/total memory of the first NVIDIA GPU, refreshed at most every 5 seconds."""
    if time.monotonic() - _gpu_cache["at"] < 5:
        return _gpu_cache["value"]
    value = None
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        name, free, total = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
        value = {"name": name, "free_mb": int(free), "total_mb": int(total)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        value = None
    _gpu_cache.update(at=time.monotonic(), value=value)
    return value


class LocalAsrEngine:
    """Owns one worker process that keeps the model on the GPU for the duration of a job."""

    def __init__(self, folder, batch_size=4, python=GPU_PYTHON, worker=WORKER, models=MODELS, log_dir=None):
        self.folder, self.batch_size = folder, batch_size
        self.python, self.worker, self.models = Path(python), Path(worker), Path(models)
        self.log_dir = Path(log_dir or Path(os.environ.get("LOCALAPPDATA", ROOT)) / "ASR-Web" / "logs")
        self.process, self.reader, self.lines, self.counter, self.info = None, None, queue.Queue(), 0, {}

    def _read(self):
        for raw in self.process.stdout:
            line = raw.decode("utf-8", "replace").strip()
            if line.startswith("{"):
                try:
                    self.lines.put(json.loads(line))
                except ValueError:
                    pass
        self.lines.put(None)

    def _wait(self, predicate, timeout, cancel):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel.is_set():
                self.close(force=True)
                check_cancel(cancel)
            try:
                message = self.lines.get(timeout=0.2)
            except queue.Empty:
                continue
            if message is None:
                raise UserError("本地识别进程意外退出，请查看日志 " + str(self.log_dir / "local-asr.log"))
            if message.get("fatal"):
                raise UserError(message["fatal"])
            if predicate(message):
                return message
        self.close(force=True)
        raise UserError("本地识别等待超时，已停止本地模型。")

    def start(self, cancel, progress):
        ok, reason = local_status(self.folder, self.python, self.models)
        if not ok:
            raise UserError(reason + "，请改用云端识别或重新安装本地 GPU 组件。")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        progress("正在把本地识别模型加载到显卡（约 20～60 秒）")
        with (self.log_dir / "local-asr.log").open("ab") as log:
            self.process = subprocess.Popen(
                [str(self.python), "-s", "-u", str(self.worker), str(self.models / self.folder)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, cwd=str(self.worker.parent), env=env,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()
        self.info = self._wait(lambda m: m.get("ready"), 600, cancel)
        progress(f"本地模型已加载：{self.info.get('gpu', 'GPU')}，剩余显存约 {self.info.get('free_after_mb', '?')} MB")
        return self.info

    def _request(self, audios, language, cancel):
        if self.process is None or self.process.poll() is not None:
            raise UserError("本地识别进程未运行。")
        self.counter += 1
        ident = self.counter
        line = json.dumps({"id": ident, "language": language,
                           "audios": [base64.b64encode(a).decode("ascii") for a in audios]}) + "\n"
        try:
            self.process.stdin.write(line.encode("utf-8"))
            self.process.stdin.flush()
        except OSError:
            raise UserError("无法把音频发送给本地识别进程。") from None
        return self._wait(lambda m: m.get("id") == ident, 900, cancel)

    def transcribe_batch(self, audios, language, cancel):
        texts, position = [], 0
        while position < len(audios):
            chunk = audios[position:position + self.batch_size]
            response = self._request(chunk, language, cancel)
            if response.get("oom"):
                if len(chunk) == 1:
                    raise UserError("显存不足：请关闭占用显卡的软件（浏览器、游戏启动器等），"
                                    "或改用 0.6B 本地模型 / 云端识别。")
                # Retry the same audio with a smaller batch; later batches keep the smaller size.
                self.batch_size = max(1, len(chunk) // 2)
                continue
            texts.extend(self._texts(response, len(chunk)))
            position += len(chunk)
        return texts

    @staticmethod
    def _texts(response, expected):
        if response.get("error"):
            raise UserError("本地识别失败：" + str(response["error"]))
        texts = response.get("texts")
        if not isinstance(texts, list) or len(texts) != expected or not all(isinstance(t, str) for t in texts):
            raise UserError("本地识别返回的结果数量不正确。")
        return texts

    def close(self, force=False):
        """Stop the worker and free GPU memory; force skips the graceful shutdown (cancel, timeout)."""
        process, self.process = self.process, None
        if process is None:
            return
        try:
            process.stdin.close()
        except OSError:
            pass
        if force:
            process.kill()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        if self.reader is not None:
            self.reader.join(timeout=5)
        process.stdout.close()
