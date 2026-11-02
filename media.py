from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread
import math
import os
import secrets
import subprocess
import tempfile
import time
import wave

import av
import imageio_ffmpeg
import numpy as np
import onnxruntime as ort

from common import Cancelled, UserError, check_cancel

ROOT = Path(__file__).resolve().parent
VAD_PATH = ROOT / "assets" / "silero_vad_16k.onnx"
SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}
VIDEO_EXTENSIONS = {".mkv", ".mp4", ".mov", ".avi", ".webm", ".m4v", ".ts", ".flv", ".wmv"}


@dataclass(frozen=True)
class Media:
    id: str
    path: Path
    duration: float
    size: int
    modified_ns: int
    kind: str

    def public(self):
        return {"id": self.id, "name": self.path.name, "filename": self.path.name,
                "path": str(self.path), "duration_seconds": round(self.duration, 3),
                "size_bytes": self.size, "kind": self.kind, "output_directory": str(self.path.parent)}

    def unchanged(self):
        info = self.path.stat()
        return info.st_size == self.size and info.st_mtime_ns == self.modified_ns


def probe_media(path, ident=None):
    raw = str(path).strip().strip('"')
    if not raw or "://" in raw or raw.startswith(("\\\\", "//")):
        raise UserError("请选择本机磁盘上的视频或音频，不支持网址或网络共享路径。")
    source = Path(raw).expanduser()
    if not source.is_absolute() or not source.is_file():
        raise UserError("文件不存在，请选择本机视频或音频。")
    source = source.resolve()
    if source.suffix.lower() not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise UserError("不支持此格式，请选择 MKV、MP4、MOV、AVI、WebM 或常见音频文件。")
    try:
        info = source.stat()
        with av.open(str(source), options={"protocol_whitelist": "file,pipe"}) as container:
            if not container.streams.audio:
                raise UserError("这个文件没有音轨，不能生成语音字幕。")
            stream = container.streams.audio[0]
            duration = container.duration / av.time_base if container.duration is not None else None
            if duration is None and stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            kind = "video" if container.streams.video else "audio"
    except UserError:
        raise
    except Exception:
        raise UserError("无法读取媒体文件，可能文件损坏、格式不受支持或正在被其他程序写入。") from None
    if duration is None or not math.isfinite(duration) or duration <= 0:
        raise UserError("无法读取音频时长，请换一个可正常播放的文件。")
    if duration > 24 * 3600:
        raise UserError("暂不处理超过 24 小时的单个文件。")
    return Media(ident or secrets.token_hex(16), source, duration, info.st_size, info.st_mtime_ns, kind)


def video_size(path):
    """Display size of the first video stream, or None for audio files and unreadable media."""
    try:
        with av.open(str(path), options={"protocol_whitelist": "file,pipe"}) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            width, height = stream.width, stream.height
            if stream.sample_aspect_ratio and stream.sample_aspect_ratio > 0:
                width = width * stream.sample_aspect_ratio
            return (float(width), float(height)) if width and height else None
    except Exception:
        return None


class MediaLibrary:
    def __init__(self):
        self.lock = Lock()
        self.items = {}

    def add(self, path, ident=None):
        media = probe_media(path, ident)
        with self.lock:
            self.items[media.id] = media
        return media

    def get(self, ident):
        with self.lock:
            media = self.items.get(ident)
        if media is None:
            raise UserError("所选文件已失效，请重新选择视频。")
        return media


_dialog_lock = Lock()


def choose_path(directory=False):
    if not _dialog_lock.acquire(blocking=False):
        raise UserError("文件选择窗口已经打开，请先完成或取消选择。")
    root = None
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        if directory:
            result = filedialog.askdirectory(parent=root, title="选择字幕保存文件夹", mustexist=True)
        else:
            result = filedialog.askopenfilename(parent=root, title="选择视频或音频（视频不会上传云端）", filetypes=[
                ("视频和音频", "*.mkv *.mp4 *.mov *.avi *.webm *.m4v *.ts *.flv *.wmv *.mp3 *.wav *.m4a *.aac *.flac *.ogg *.opus"),
                ("所有文件", "*.*"),
            ])
        return result or None
    finally:
        if root is not None:
            root.destroy()
        _dialog_lock.release()


def clip_range(media, start=0, duration=None):
    if not math.isfinite(start) or start < 0 or start >= media.duration:
        raise UserError("起始时间必须在视频时长范围内。")
    remaining = media.duration - start
    if duration is None:
        duration = remaining
    if not math.isfinite(duration) or duration <= 0:
        raise UserError("处理时长必须大于 0。")
    return start, min(duration, remaining)


def extract_audio(media, output_path, start, duration, progress, cancel):
    check_cancel(cancel)
    if not media.unchanged():
        raise UserError("文件在选择后发生变化，请重新选择，避免使用过期缓存。")
    command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-nostdin", "-n",
               "-ss", f"{start:.6f}", "-protocol_whitelist", "file,pipe", "-i", str(media.path),
               "-t", f"{duration:.6f}", "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
               "-af", "aresample=async=1:first_pts=0", "-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", str(output_path)]
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=errors, creationflags=flags)
        def read_progress():
            for raw in process.stdout:
                line = raw.decode("ascii", "ignore").strip()
                if line.startswith("out_time_us="):
                    try:
                        seconds = min(float(line.split("=", 1)[1]) / 1000000, duration)
                        progress(seconds / duration, f"提取音频：{seconds:.0f} / {duration:.0f} 秒")
                    except ValueError:
                        pass
        reader = Thread(target=read_progress, daemon=True)
        reader.start()
        try:
            while process.poll() is None:
                if cancel.wait(0.1):
                    process.terminate()
                    try:
                        process.wait(5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    raise Cancelled("已取消本地音频提取，尚未发送的语音不会上传。")
            reader.join(timeout=3)
            if process.returncode:
                raise UserError("本机 FFmpeg 音频提取失败，请确认文件可以正常播放且音轨未加密。")
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            reader.join(timeout=3)
            process.stdout.close()
    with wave.open(str(output_path), "rb") as audio:
        actual = audio.getnframes() / audio.getframerate()
        if actual <= 0:
            raise UserError("提取后的音频为空。")
    progress(1, f"音频提取完成：{actual:.1f} 秒")
    return actual


def probabilities_to_segments(probs, duration, threshold=0.5, silence_seconds=0.28, max_seconds=8.0,
                              merge_gap=1.0, merge_max_seconds=10.0):
    step = FRAME_SAMPLES / SAMPLE_RATE
    silence_frames = max(1, round(silence_seconds / step))
    raw, speech_start, silence_start = [], None, None
    for index, probability in enumerate(probs):
        if probability >= threshold:
            if speech_start is None:
                speech_start = index
            silence_start = None
        elif speech_start is not None and probability < threshold - 0.15:
            if silence_start is None:
                silence_start = index
            if index - silence_start >= silence_frames:
                if (silence_start - speech_start) * step >= 0.16:
                    raw.append((speech_start * step, min(silence_start * step, duration)))
                speech_start = silence_start = None
    if speech_start is not None and duration - speech_start * step >= 0.16:
        raw.append((speech_start * step, duration))
    split = []
    for start, end in raw:
        while end - start > max_seconds:
            left = max(0, math.ceil((start + max_seconds - 2) / step))
            right = min(len(probs), math.floor((start + max_seconds) / step))
            if right > left:
                # Pick a real low-speech-probability frame, not a text-proportional timestamp.
                cut_index = min(range(left, right), key=lambda i: (probs[i], -i))
                cut = min(max(cut_index * step, start + 1), end)
            else:
                cut = start + max_seconds
            split.append([start, cut])
            start = cut
        if end - start >= 0.12:
            split.append([start, end])
    merged = []
    for start, end in split:
        # Join pauses inside a sentence so recognition and translation see whole phrases.
        if merged and start - merged[-1][1] <= merge_gap and end - merged[-1][0] <= merge_max_seconds:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    padded = [[max(0, s - 0.08), min(duration, e + 0.1)] for s, e in merged]
    for index in range(len(padded) - 1):
        if padded[index][1] > padded[index + 1][0]:
            midpoint = (merged[index][1] + merged[index + 1][0]) / 2
            padded[index][1] = midpoint
            padded[index + 1][0] = midpoint
    return [{"start": round(s, 6), "end": round(e, 6)} for s, e in padded if e > s]


class SpeechDetector:
    def detect(self, wav_path, progress, cancel):
        if not VAD_PATH.is_file():
            raise UserError("本机语音检测模型缺失，请恢复 assets/silero_vad_16k.onnx。")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(VAD_PATH), sess_options=options, providers=["CPUExecutionProvider"])
        state = np.zeros((2, 1, 128), dtype=np.float32)
        context = np.zeros((1, 64), dtype=np.float32)
        probabilities = []
        with wave.open(str(wav_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
                raise UserError("本地音频格式不符合语音检测要求。")
            total = audio.getnframes()
            duration = total / SAMPLE_RATE
            frames = math.ceil(total / FRAME_SAMPLES)
            for index in range(frames):
                check_cancel(cancel)
                data = audio.readframes(FRAME_SAMPLES)
                samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
                if len(samples) < FRAME_SAMPLES:
                    samples = np.pad(samples, (0, FRAME_SAMPLES - len(samples)))
                chunk = np.concatenate((context, samples.reshape(1, -1)), axis=1)
                output, state = session.run(None, {"input": chunk, "state": state, "sr": np.array(SAMPLE_RATE, dtype=np.int64)})
                probability = float(output.reshape(-1)[0])
                if not math.isfinite(probability):
                    raise UserError("语音检测模型返回了无效结果。")
                probabilities.append(probability)
                context = chunk[:, -64:]
                if index % max(1, frames // 100) == 0:
                    progress((index + 1) / frames, f"检测语音：{min((index + 1) * FRAME_SAMPLES / SAMPLE_RATE, duration):.0f} / {duration:.0f} 秒")
        segments = probabilities_to_segments(probabilities, duration)
        progress(1, f"语音检测完成：按停顿合并为 {len(segments)} 句")
        return segments


def read_wav_segment(path, start, end):
    with wave.open(str(path), "rb") as source:
        rate = source.getframerate()
        first = max(0, round(start * rate))
        last = min(source.getnframes(), round(end * rate))
        if last <= first:
            raise UserError("语音片段长度无效。")
        source.setpos(first)
        pcm = source.readframes(last - first)
    output = BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(rate)
        target.writeframes(pcm)
    return output.getvalue()
