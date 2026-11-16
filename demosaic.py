"""Mosaic removal with Lada (AGPL-3.0), run as the separate program lada-cli.exe (installed by setup_lada.py).

No Lada code is imported: the app starts the command line program, reads its progress bar and checks the video it writes.
The result is AI inpainting of the pixelated areas, not a recovery of the original picture.
"""
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
import os
import re
import secrets
import shutil
import subprocess
import tempfile
import time

import av
import imageio_ffmpeg

from common import Cancelled, UserError, safe_message
from local_asr import GPU_GATE
from subtitles import unused_path

ROOT = Path(__file__).resolve().parent
LADA_CLI = ROOT / "lada" / "lada-cli.exe"
ENCODING_PRESET = "hevc-nvidia-gpu-hq"
SAMPLE_SECONDS = 60
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
LOG_NAME = "demosaic.log"
_PERCENT = re.compile(r"(\d{1,3})%\|")
_FRAMES = re.compile(r"\((\d+)f\)")
_SPEED = re.compile(r"Speed:\s*([\d.]+)fps")
_REMAINING = re.compile(r"Remaining:\s*([\d:]+)")
# The bar always ends with the speed; Lada prints errors right after it without a line break.
_BAR_END = re.compile(r"Speed:\s*(?:[\d.]+fps|\?)")


def lada_status(cli=LADA_CLI):
    if not Path(cli).is_file():
        return False, "去马赛克组件未安装：请在程序目录运行 python setup_lada.py"
    return True, ""


def parse_progress(line):
    """One refresh of Lada's progress bar (it is run with LANGUAGE=en); None for any other output."""
    percent, frames = _PERCENT.search(line), [int(n) for n in _FRAMES.findall(line)]
    if not percent or not frames:
        return None
    speed, remaining = _SPEED.search(line), _REMAINING.search(line)
    return {"percent": min(int(percent.group(1)), 100), "frames_done": frames[0],
            "frames_total": frames[0] + frames[1] if len(frames) > 1 else 0,
            "fps": float(speed.group(1)) if speed else 0.0, "remaining": remaining.group(1) if remaining else ""}


def progress_label(info):
    label = (f"去马赛克：{info['frames_done']} / {info['frames_total']} 帧" if info["frames_total"]
             else f"去马赛克：已处理 {info['frames_done']} 帧")
    if info["fps"]:
        label += f" · {info['fps']:.1f} 帧/秒"
    if info["remaining"]:
        label += f" · 预计还需 {info['remaining']}"
    return label


def restored_name(source, sample_start=None):
    stem = Path(source).stem
    if sample_start is None:
        return f"{stem}.restored.mp4"
    seconds = int(sample_start)
    return f"{stem}.restored.sample-{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s.mp4"


def kill_tree(process):
    """Lada starts FFmpeg and worker processes of its own, so the whole tree is stopped."""
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, creationflags=NO_WINDOW, timeout=30)
    else:
        process.kill()
    try:
        process.wait(15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def check_video(path, expected_duration, needs_audio):
    try:
        with av.open(str(path), options={"protocol_whitelist": "file,pipe"}) as container:
            has_video, has_audio = bool(container.streams.video), bool(container.streams.audio)
            duration = container.duration / av.time_base if container.duration is not None else None
    except Exception:
        raise UserError("去马赛克生成的视频无法读取，可能处理中断或磁盘已满。") from None
    if not has_video:
        raise UserError("去马赛克生成的文件里没有视频画面。")
    if needs_audio and not has_audio:
        raise UserError("去马赛克生成的视频缺少音轨（合并原音轨失败），未保存。")
    if duration is not None and expected_duration and duration < expected_duration - max(2.0, expected_duration * 0.01):
        raise UserError(f"去马赛克生成的视频只有 {duration:.0f} 秒，比原视频短，可能处理中断，未保存。")


class MosaicRemover:
    def __init__(self, cli=LADA_CLI, command=None, gate=GPU_GATE, log_dir=None, preset=ENCODING_PRESET):
        self.cli = Path(cli)
        # Tests replace the program with a fake one run by the Python interpreter.
        self.command = list(command) if command else [str(self.cli)]
        self.gate, self.preset = gate, preset
        self.log_dir = Path(log_dir or Path(os.environ.get("LOCALAPPDATA", ROOT)) / "ASR-Web" / "logs")

    def status(self):
        return lada_status(self.cli)

    def run(self, media, sample_start, update, cancel):
        ok, reason = self.status()
        if not ok:
            raise UserError(reason + "。")
        if not media.unchanged():
            raise UserError("视频在选择后发生变化，请重新选择。")
        source = media.path
        seconds = media.duration if sample_start is None else min(SAMPLE_SECONDS, media.duration - sample_start)
        # Lada writes the new picture to a temporary file and then muxes it with the audio, so both exist for a while.
        # The result was about 0.75x the source size in tests (H.265, NVENC high quality); 2x leaves room for busy scenes.
        needed = media.size * seconds / media.duration * 2 + 2 ** 30
        free = shutil.disk_usage(source.parent).free
        if free < needed:
            raise UserError(f"视频所在磁盘只剩 {free / 2 ** 30:.1f} GB，去马赛克约需 {needed / 2 ** 30:.1f} GB "
                            "临时和输出空间，请先释放空间或缩短处理范围。")
        try:
            # Next to the video, so the finished file is moved instead of copied and nothing fills another drive.
            work = Path(tempfile.mkdtemp(prefix=".demosaic-", dir=source.parent))
        except OSError:
            raise UserError("无法在视频所在文件夹写入临时文件，请检查文件夹权限。") from None
        try:
            if sample_start is not None:
                update(stage=f"正在截取试处理片段（{seconds:.0f} 秒）", progress=0)
                source = self.cut(source, work / "sample.mkv", sample_start, seconds, cancel)
            output = work / "restored.mp4"
            self.restore(source, output, work, update, cancel)
            update(stage="正在检查生成的视频", progress=99)
            check_video(output, seconds, needs_audio=True)
            final = unused_path(media.path.parent / restored_name(media.path, sample_start))
            os.replace(output, final)
            return final
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def cut(self, source, target, start, seconds, cancel):
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                   "-ss", f"{start:.3f}", "-protocol_whitelist", "file,pipe", "-i", str(source), "-t", f"{seconds:.3f}",
                   "-map", "0:v:0", "-map", "0:a:0", "-c", "copy", str(target)]
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=errors,
                                       creationflags=NO_WINDOW)
            try:
                while process.poll() is None:
                    if cancel.wait(0.2):
                        raise Cancelled("已取消去马赛克，未保存视频。")
            finally:
                kill_tree(process)
            errors.seek(0)
            detail = errors.read().decode("utf-8", "replace").strip()[-300:]
        if process.returncode or not target.is_file():
            raise UserError("无法截取试处理片段" + (f"：{detail}" if detail else "，请确认视频可以正常播放。"))
        return target

    def restore(self, source, output, work, update, cancel):
        command = self.command + ["--input", str(source), "--output", str(output), "--temporary-directory", str(work),
                                  "--encoding-preset", self.preset, "--detect-face-mosaics"]
        env = os.environ.copy()
        env.update(LANGUAGE="en", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        tail = deque(maxlen=8)
        self.gate.acquire("去马赛克", cancel, lambda holder: update(
            stage=f"等待显卡空闲：{holder}正在使用显卡，完成后自动开始去马赛克"))
        try:
            update(stage="正在把去马赛克模型加载到显卡（约 10～60 秒）", progress=1)
            with (self.log_dir / LOG_NAME).open("ab") as log:
                log.write(f"\n=== {datetime.now().isoformat(timespec='seconds')} {source}\n".encode("utf-8"))
                log.flush()
                try:
                    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                               stderr=subprocess.STDOUT, cwd=str(self.cli.parent), env=env,
                                               creationflags=NO_WINDOW)
                except OSError:
                    raise UserError("无法启动去马赛克程序，请重新运行 setup_lada.py 安装。") from None
                reader = Thread(target=self._read, args=(process, log, tail, update), daemon=True)
                reader.start()
                try:
                    while process.poll() is None:
                        if cancel.wait(0.2):
                            raise Cancelled("已取消去马赛克，未保存视频。")
                finally:
                    kill_tree(process)
                    reader.join(timeout=10)
                    process.stdout.close()
        finally:
            self.gate.release()
        detail = " / ".join(tail)[-400:] or f"详见日志 {self.log_dir / LOG_NAME}"
        if process.returncode:
            raise UserError(f"去马赛克程序出错（退出码 {process.returncode}）：{detail}")
        # Lada exits with code 0 even when the export failed, so the file decides.
        if not output.is_file() or output.stat().st_size == 0:
            raise UserError(f"去马赛克没有生成视频：{detail}")

    @staticmethod
    def _read(process, log, tail, update):
        def handle(raw):
            text = raw.decode("utf-8", "replace").strip()
            if not text:
                return
            info = parse_progress(text)
            if info is not None:
                update(stage=progress_label(info), progress=round(1 + info["percent"] * 0.97, 1),
                       frames_done=info["frames_done"], frames_total=info["frames_total"], fps=info["fps"],
                       remaining=info["remaining"])
                end = _BAR_END.search(text)
                text = text[end.end():].strip() if end else ""
                if not text:
                    return
            try:
                log.write((text + "\n").encode("utf-8"))
                log.flush()
            except (OSError, ValueError):
                pass
            tail.append(text)
            if text.startswith("Processing audio"):
                update(stage="正在合并原音轨", progress=98.5)

        buffer = b""
        # The progress bar redraws itself with carriage returns, so both \r and \n end a line.
        while chunk := process.stdout.read1(65536):
            buffer += chunk
            *lines, buffer = re.split(rb"[\r\n]", buffer)
            for raw in lines:
                handle(raw)
        handle(buffer)


@dataclass
class DemosaicJob:
    id: str
    filename: str
    source_path: str
    started_at: str
    sample_start: float | None = None
    sample_seconds: float = 0
    status: str = "running"
    stage: str = "正在准备去马赛克"
    progress: float = 0
    frames_done: int = 0
    frames_total: int = 0
    fps: float = 0
    remaining: str = ""
    output_path: str = ""
    subtitle_files: list = field(default_factory=list)
    error: str = ""
    seconds: float = 0
    cancel: Event = field(default_factory=Event, repr=False)
    clock: float = field(default_factory=time.monotonic, repr=False)

    def public(self):
        elapsed = time.monotonic() - self.clock if self.status == "running" else self.seconds
        return {"id": self.id, "filename": self.filename, "source_path": self.source_path, "started_at": self.started_at,
                "sample": self.sample_start is not None, "sample_start": self.sample_start,
                "sample_seconds": round(self.sample_seconds, 1),
                "status": self.status, "stage": self.stage, "progress": self.progress, "frames_done": self.frames_done,
                "frames_total": self.frames_total, "fps": self.fps, "remaining": self.remaining,
                "output_path": self.output_path, "output_name": Path(self.output_path).name if self.output_path else "",
                "subtitle_files": list(self.subtitle_files), "error": self.error, "elapsed_seconds": round(elapsed, 1),
                "cancel_requested": self.cancel.is_set()}


class DemosaicStore:
    def __init__(self, remover=None, on_done=None):
        self.remover = remover if remover is not None else MosaicRemover()
        # Called with (source path, restored path) after a whole video was restored; returns the subtitle copies made.
        self.on_done = on_done
        self.lock = Lock()
        self.jobs = OrderedDict()

    def status(self):
        return self.remover.status()

    def start(self, media, sample_start=None):
        with self.lock:
            if any(job.status == "running" for job in self.jobs.values()):
                raise UserError("已有去马赛克任务正在处理，请等待完成或先取消。")
            job = DemosaicJob(secrets.token_hex(16), media.path.name, str(media.path),
                              datetime.now(timezone.utc).isoformat(), sample_start,
                              min(SAMPLE_SECONDS, media.duration - sample_start) if sample_start is not None else 0)
            self.jobs[job.id] = job
            while len(self.jobs) > 10:
                self.jobs.popitem(last=False)
        Thread(target=self.run, args=(job, media), daemon=True).start()
        return job

    def run(self, job, media):
        def update(**changes):
            with self.lock:
                for key, value in changes.items():
                    if key in DemosaicJob.__dataclass_fields__ and key not in {"id", "cancel", "clock", "status"}:
                        setattr(job, key, value)
        status = "failed"
        try:
            output = self.remover.run(media, job.sample_start, update, job.cancel)
            update(output_path=str(output), progress=100,
                   stage="试处理完成，已保存为新视频" if job.sample_start is not None else "去马赛克完成，已保存为新视频")
            status = "done"
        except Cancelled as exc:
            status = "cancelled"
            update(error=safe_message(exc), stage="已取消，未保存去码视频")
        except UserError as exc:
            update(error=safe_message(exc), stage="去马赛克失败")
        except Exception:
            update(error="本机去马赛克发生异常，请检查磁盘空间，或查看日志 %LOCALAPPDATA%\\ASR-Web\\logs\\" + LOG_NAME,
                   stage="去马赛克失败")
        if status == "done" and job.sample_start is None and self.on_done is not None:
            try:
                update(subtitle_files=self.on_done(job.source_path, job.output_path))
            except (OSError, UserError):
                update(error="去码视频已保存，但没能为它复制字幕文件。")
        with self.lock:
            job.seconds = time.monotonic() - job.clock
            job.status = status

    def get(self, ident):
        with self.lock:
            return self.jobs.get(ident)

    def latest(self):
        with self.lock:
            return next(reversed(self.jobs.values()), None)

    def latest_output(self, source_path):
        """The newest whole-video result of a source that still exists."""
        with self.lock:
            for job in reversed(self.jobs.values()):
                if (job.status == "done" and job.sample_start is None and job.source_path == source_path
                        and Path(job.output_path).is_file()):
                    return job.output_path
        return None
