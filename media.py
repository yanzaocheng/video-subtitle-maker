from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Event, Lock, Thread
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
from firered_vad import FireRedVad, to_grid

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


class AnyEvent:
    """A cancel signal that is set when any of several events is."""

    def __init__(self, *events):
        self.events = events

    def is_set(self):
        return any(event.is_set() for event in self.events)


class SpeechDetector:
    """Finds the sentences of a 16 kHz mono WAV file with two models, Silero VAD and FireRedVAD.

    Silero hears the file twice. Once as one stream: the model carries a state from frame to frame. Long files are
    cut into parts that run on several threads; each part first replays the minutes before it, because the state
    needs that long to match an uninterrupted run (measured: 10 minutes give the same sentences as one stream).
    That long memory has a price: after clear speech the model stops reporting speech that is quieter or covered by
    noise (a shower, music) for minutes on end, and it cuts sentences short. So the file is heard a second time in
    short windows that each start from an empty state, and a frame is speech when either pass says so.

    FireRedVAD hears more still, so when its weights are there the sentences are cut where it hears speech, and
    Silero says how sure that is. Measured against the burned-in Chinese subtitles of a 2-hour film: lines without
    any subtitle of ours went from 21% to 10%, lines whose meaning was recognised from 27% to 31%.

    A sentence that Silero does not support (only Silero's second pass, or only FireRedVAD, heard it) comes back
    with "recovered": True: it is less certain (often a moan or noise), and the caller is stricter with it.
    """
    CONTEXT = 64  # samples of the previous frame the model sees in front of each frame
    BLOCK = 500  # frames read and analysed between two looks at the cancel signal
    PART_SECONDS = 1800
    WARMUP_SECONDS = 600
    WINDOW_SECONDS = 10  # second pass: length of a window; they overlap by half and the higher probability counts
    THRESHOLD = 0.5  # probability from which a frame is speech

    def __init__(self, workers=None, firered=None):
        self.workers = workers if workers is not None else max(1, min(6, (os.cpu_count() or 2) // 2))
        self.session, self.lock = None, Lock()
        # False: Silero alone (as before FireRedVAD was added).
        self.firered = FireRedVad() if firered is None else firered

    def load(self):
        with self.lock:
            if self.session is None:
                if not VAD_PATH.is_file():
                    raise UserError("本机语音检测模型缺失，请恢复 assets/silero_vad_16k.onnx。")
                options = ort.SessionOptions()
                options.intra_op_num_threads = 1
                options.inter_op_num_threads = 1
                self.session = ort.InferenceSession(str(VAD_PATH), sess_options=options, providers=["CPUExecutionProvider"])
            return self.session

    def stream(self, session, wav_path, first, last, warmup, advance, cancel):
        """Speech probability of the frames first..last; the model state starts `warmup` frames before them."""
        begin, context = first - warmup, self.CONTEXT
        probabilities = np.empty(last - first, dtype=np.float32)
        feed = {"input": None, "state": np.zeros((2, 1, 128), dtype=np.float32), "sr": np.array(SAMPLE_RATE, dtype=np.int64)}
        with wave.open(str(wav_path), "rb") as audio:
            tail = np.zeros(context, dtype=np.float32)
            if begin > 0:
                audio.setpos(begin * FRAME_SAMPLES - context)
                tail = np.frombuffer(audio.readframes(context), dtype="<i2").astype(np.float32) / 32768.0
            for position in range(begin, last, self.BLOCK):
                check_cancel(cancel)
                count = min(self.BLOCK, last - position)
                samples = np.frombuffer(audio.readframes(count * FRAME_SAMPLES), dtype="<i2").astype(np.float32) / 32768.0
                # The last frame of the file is padded with silence.
                block = np.zeros(context + count * FRAME_SAMPLES, dtype=np.float32)
                block[:context] = tail
                block[context:context + len(samples)] = samples
                for index in range(count):
                    offset = index * FRAME_SAMPLES
                    feed["input"] = block[offset:offset + context + FRAME_SAMPLES].reshape(1, -1)
                    output, feed["state"] = session.run(None, feed)
                    if position + index >= first:
                        probabilities[position + index - first] = output[0, 0]
                tail = block[-context:].copy()
                advance(count)
        return probabilities

    def detect(self, wav_path, progress, cancel):
        session = self.load()
        with wave.open(str(wav_path), "rb") as audio:
            if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, SAMPLE_RATE):
                raise UserError("本地音频格式不符合语音检测要求。")
            total = audio.getnframes()
        duration = total / SAMPLE_RATE
        frames = math.ceil(total / FRAME_SAMPLES)
        parts = max(1, min(self.workers, int(duration // self.PART_SECONDS)))
        size = max(1, math.ceil(frames / parts))
        warmup = round(self.WARMUP_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
        spans = [(first, min(frames, first + size), min(warmup, first)) for first in range(0, frames, size)]
        firered = self.firered if self.firered and self.firered.available() else None
        # Each pass is an equal share of the progress bar.
        passes = 3 if firered else 2
        results = self.analyse(session, wav_path, spans, cancel, lambda ratio: progress(
            ratio / passes, f"检测语音：{ratio * duration:.0f} / {duration:.0f} 秒"))
        stream = np.concatenate(results) if results else np.empty(0, dtype=np.float32)
        fresh = self.windows(session, wav_path, frames, cancel, lambda ratio: progress(
            (1 + ratio) / passes, f"复查低声和被噪声盖住的语音：{ratio * duration:.0f} / {duration:.0f} 秒"))
        silero = np.maximum(stream, fresh)
        if firered:
            heard = to_grid(firered.probabilities(wav_path, self.workers, cancel, lambda ratio: progress(
                (2 + ratio) / passes, f"用 FireRedVAD 再听一遍：{ratio * duration:.0f} / {duration:.0f} 秒")),
                frames, FRAME_SAMPLES)
            sure = silero
        else:
            heard, sure = silero, stream
        if not (np.isfinite(heard).all() and np.isfinite(sure).all()):
            raise UserError("语音检测模型返回了无效结果。")
        segments = probabilities_to_segments(heard.tolist(), duration, threshold=self.THRESHOLD)
        step = FRAME_SAMPLES / SAMPLE_RATE
        for item in segments:
            first = int(item["start"] / step)
            if not (sure[first:max(first + 1, math.ceil(item["end"] / step))] >= self.THRESHOLD).any():
                item["recovered"] = True
        recovered = sum(1 for item in segments if item.get("recovered"))
        label = f"语音检测完成：按停顿合并为 {len(segments)} 句"
        if recovered:
            label += f"，其中 {recovered} 句是复查时才找到的（低声、被噪声盖住，也可能只是语气声）"
        progress(1, label)
        return segments

    def analyse(self, session, wav_path, spans, cancel, report):
        """The probabilities of each span (first frame, last frame, warm-up frames), computed on several threads."""
        work, done, lock = sum(last - first + warm for first, last, warm in spans), [0], Lock()
        # The other spans stop as soon as one fails or the job is cancelled; the first error is the one reported.
        failed, errors = Event(), []
        stop = AnyEvent(cancel, failed)

        def advance(count):
            with lock:
                done[0] += count
                ratio = done[0] / work
            report(ratio)

        def part(span):
            try:
                return self.stream(session, wav_path, *span, advance, stop)
            except BaseException as exc:
                with lock:
                    if not failed.is_set():
                        errors.append(exc)
                    failed.set()

        threads = min(self.workers, len(spans))
        if threads <= 1:
            results = [part(span) for span in spans]
        else:
            with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="speech-detect") as executor:
                results = list(executor.map(part, spans))
        if errors:
            raise errors[0]
        return results

    def windows(self, session, wav_path, frames, cancel, report):
        """The probabilities of the second pass: half-overlapping windows that each start from an empty state."""
        size = max(2, round(self.WINDOW_SECONDS * SAMPLE_RATE / FRAME_SAMPLES))
        spans = [(first, min(frames, first + size), 0) for first in range(0, frames, size // 2)]
        probabilities = np.zeros(frames, dtype=np.float32)
        for (first, last, _), part in zip(spans, self.analyse(session, wav_path, spans, cancel, report)):
            np.maximum(probabilities[first:last], part, out=probabilities[first:last])
        return probabilities


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
