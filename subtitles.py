from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from threading import Event, Lock
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import unicodedata

from common import (ASR_ENGINES, PROVIDERS, Cancelled, LANGUAGES, MODEL, TRANSLATION_MODEL, UserError, check_cancel,
                    is_nonverbal, language_matches)
from local_asr import LocalAsrEngine
from media import SpeechDetector, extract_audio, read_wav_segment, video_size

ROOT = Path(__file__).resolve().parent
TIME_AXIS_NOTE = "本地 Silero VAD 按停顿合并的整句时间轴（每条最长约 10 秒），不是逐字强制对齐。低声、重叠对白需试听校对。"
TRANSLATION_BATCH = 20
CLOSING_PUNCTUATION = set("。，、！？；：」』）】〉》…,.!?;:)]}")

# One subtitle style drives both the editor preview and the exported .ass files.
# Sizes and margins are pixels at 1080p video height; "size" is the em size of the text.
SUBTITLE_FONTS = {
    # family: (label, cell height / em). VSFilter and libass size fonts by winAscent + winDescent.
    "Microsoft YaHei": ("微软雅黑", 1.32),
    "DengXian": ("等线", 1.042),
    "SimHei": ("黑体", 1.0),
    "KaiTi": ("楷体", 1.0),
    "SimSun": ("宋体", 1.0),
    "FangSong": ("仿宋", 1.0),
}
DEFAULT_STYLE = {"font": "Microsoft YaHei", "size": 54, "color": "#FFFFFF", "outline_color": "#000000",
                 "border": "outline", "position": "bottom", "margin": 50}
STYLE_CHOICES = {"font": set(SUBTITLE_FONTS), "border": {"outline", "box", "none"}, "position": {"bottom", "top"}}
STYLE_RANGES = {"size": (24, 120), "margin": (0, 400)}
ORIGINAL_SCALE = 0.75  # the original line under the translation in bilingual subtitles
OUTLINE_EM = 0.055
BOX_PADDING_EM = 0.12
BOX_ALPHA = 0x66  # about 60 % opaque
SIDE_MARGIN = 0.05  # share of the video width kept free on the left and right


def srt_time(seconds):
    if not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise UserError("字幕时间戳无效。")
    milliseconds = round(seconds * 1000)
    hours, milliseconds = divmod(milliseconds, 3600000)
    minutes, milliseconds = divmod(milliseconds, 60000)
    seconds, milliseconds = divmod(milliseconds, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def clean_text(text):
    return " ".join(re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text).split()).strip()


def wrap_subtitle(text, width=48):
    # Line breaks typed in the editor are kept; each line is wrapped on its own.
    lines = []
    for paragraph in str(text).split("\n"):
        lines.extend(_wrap_line(clean_text(paragraph), width))
    return "\n".join(lines)


def _wrap_line(text, width):
    lines, current, used = [], "", 0
    for token in re.findall(r"[A-Za-z0-9']+\s*|.", text):
        size = sum(2 if unicodedata.east_asian_width(c) in {"W", "F"} else 1 for c in token)
        # Keep closing punctuation on the previous line instead of starting a line with it.
        if current and used + size > width and token not in CLOSING_PUNCTUATION:
            lines.append(current.strip())
            current, used = "", 0
        current += token
        used += size
    if current.strip():
        lines.append(current.strip())
    return lines


def normalize_style(value, strict=True):
    """Validate a subtitle style. Lenient mode replaces bad fields with defaults (for the saved preference)."""
    if not isinstance(value, dict):
        if strict:
            raise UserError("字幕样式格式不正确。")
        value = {}
    style = {}
    for key, default in DEFAULT_STYLE.items():
        item = value.get(key, default)
        if key in STYLE_RANGES:
            low, high = STYLE_RANGES[key]
            valid = (isinstance(item, (int, float)) and not isinstance(item, bool) and math.isfinite(item)
                     and low <= item <= high)
            if valid:
                item = round(float(item), 1)
                item = int(item) if item.is_integer() else item
        elif key in STYLE_CHOICES:
            valid = isinstance(item, str) and item in STYLE_CHOICES[key]
        else:
            valid = isinstance(item, str) and re.fullmatch(r"#[0-9A-Fa-f]{6}", item) is not None
            item = item.upper() if valid else item
        if not valid:
            if strict:
                raise UserError("字幕样式设置无效，请点击“恢复默认样式”后重试。")
            item = default
        style[key] = item
    return style


def ass_time(seconds):
    centiseconds = round(seconds * 100)
    hours, centiseconds = divmod(centiseconds, 360000)
    minutes, centiseconds = divmod(centiseconds, 6000)
    seconds, centiseconds = divmod(centiseconds, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def ass_color(hex_color, alpha=0):
    red, green, blue = (int(hex_color[i:i + 2], 16) for i in (1, 3, 5))
    return f"&H{alpha:02X}{blue:02X}{green:02X}{red:02X}"


def ass_number(value):
    return f"{value:.1f}".rstrip("0").rstrip(".")


def ass_escape(text):
    # Braces start override tags and backslashes start escapes in ASS text.
    return text.replace("\\", "＼").replace("{", "｛").replace("}", "｝")


def make_ass(segments, style=None, size=None, bilingual=False):
    style = normalize_style(DEFAULT_STYLE if style is None else style)
    width, height = size if size and size[0] > 0 and size[1] > 0 else (1920, 1080)
    play_y = 1080
    play_x = max(1, round(play_y * width / height))
    side = round(play_x * SIDE_MARGIN)
    cell = SUBTITLE_FONTS[style["font"]][1]
    box = style["border"] == "box"

    def border(em):
        # Outline width, or the padding around the text in box mode.
        return 0 if style["border"] == "none" else em * (BOX_PADDING_EM if box else OUTLINE_EM)

    def lines(text, em):
        # Renderers wrap only at spaces, so CJK lines are broken here to fit the chosen font size.
        units = max(8, int((play_x - 2 * side - 2 * border(em)) / em * 2 * 0.96))
        return "\\N".join(ass_escape(line) for line in wrap_subtitle(text, units).split("\n"))

    em, small = style["size"], style["size"] * ORIGINAL_SCALE
    events = []
    for item in segments:
        start, end = ass_time(item["start"]), ass_time(item["end"])
        if start == end or not item["text"].strip():
            continue
        text = lines(item["text"], em)
        if bilingual and (item.get("original_text") or "").strip():
            text += f"\\N{{\\fs{ass_number(small * cell)}\\bord{ass_number(border(small))}}}" + lines(item["original_text"], small)
        events.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")
    header = [
        "[Script Info]",
        "; 由“视频字幕制作”生成，字幕样式可在客户端第 4 步修改",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: None",
        f"PlayResX: {play_x}",
        f"PlayResY: {play_y}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Default,{style['font']},{ass_number(em * cell)},{ass_color(style['color'])},&H000000FF,"
        f"{ass_color(style['outline_color'], BOX_ALPHA if box else 0)},&H00000000,0,0,0,0,100,100,0,0,{3 if box else 1},"
        f"{ass_number(border(em))},0,{8 if style['position'] == 'top' else 2},{side},{side},{round(style['margin'])},1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # The BOM keeps VSFilter-based players from reading the file in the ANSI code page.
    return "\ufeff" + "\n".join(header + events) + "\n"


def make_srt(segments, bilingual=False):
    blocks, previous_end = [], 0.0
    for item in segments:
        start, end = item["start"], item["end"]
        if start < previous_end - 0.002 or end <= start or not item["text"].strip():
            raise UserError("字幕时间轴重叠、倒序或内容为空，未导出损坏字幕。")
        rendered_start, rendered_end = srt_time(start), srt_time(end)
        if rendered_start == rendered_end:
            continue
        text = wrap_subtitle(item["text"])
        if bilingual and item.get("original_text"):
            text += "\n" + wrap_subtitle(item["original_text"])
        blocks.append(f"{len(blocks) + 1}\n{rendered_start} --> {rendered_end}\n{text}")
        previous_end = end
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def subtitle_files(language, original, translated, bilingual, style=None, size=None):
    """Render a job's subtitle files: kind -> (file name suffix, content)."""
    target = language if language != "original" else "source"
    both = bool(translated) and bilingual and language != "original"
    files = {}
    if translated:
        files["srt"] = (f"{target}.srt", make_srt(translated))
    if original and language != "original":
        files["original"] = ("source.srt", make_srt(original))
    if both:
        files["bilingual"] = (f"{language}.bilingual.srt", make_srt(translated, bilingual=True))
    if translated:
        files["ass"] = (f"{target}.ass", make_ass(translated, style, size))
    if both:
        files["bilingual_ass"] = (f"{language}.bilingual.ass", make_ass(translated, style, size, bilingual=True))
    return files


def unused_path(path):
    candidate, number = path, 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}.{number}{path.suffix}")
        number += 1
    return candidate


def rewrite_files(saved_files, language, original, translated, bilingual, metadata_update, style=None, size=None):
    """Replace a job's own subtitle files in place after the user edited them; returns the job's file list."""
    files = subtitle_files(language, original, translated, bilingual, style, size)
    saved = [dict(item) for item in saved_files]
    kinds = {item["kind"] for item in saved}
    # Jobs exported before styled subtitles existed get their .ass files next to the matching SRT.
    for kind, sibling in (("ass", "srt"), ("bilingual_ass", "bilingual")):
        base = next((Path(item["path"]) for item in saved if item["kind"] == sibling), None)
        if kind in files and kind not in kinds and base is not None:
            position = next((i for i, item in enumerate(saved) if item["kind"] == "json"), len(saved))
            saved.insert(position, {"kind": kind, "path": str(unused_path(base.with_suffix(".ass")))})
    for item in saved:
        path, kind = Path(item["path"]), item["kind"]
        if kind == "json":
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                metadata = {}
            metadata.update(metadata_update)
            content = json.dumps(metadata, ensure_ascii=False, indent=2)
        elif kind in files:
            content = files[kind][1]
        else:
            continue
        if not path.parent.is_dir():
            raise UserError("字幕所在文件夹已不存在，未能保存修改。")
        temporary = path.with_name(path.name + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
        os.replace(temporary, path)
    return saved


class ResultCache:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()

    def key(self, namespace, data):
        if not isinstance(data, bytes):
            data = json.dumps(data, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(namespace.encode() + b"\0" + data).hexdigest()

    def get(self, key):
        path = self.root / (key + ".json")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def put(self, key, value):
        with self.lock:
            path = self.root / (key + ".json")
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)


def export_files(media, language, original, translated, bilingual, output_directory, metadata, partial=False, style=None):
    directory = Path(output_directory)
    if not directory.is_dir():
        raise UserError("字幕保存目录不存在。")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", media.path.stem).rstrip(" .")[:150] or "subtitles"
    stem += ".partial" if partial else ""
    files = subtitle_files(language, original, translated, bilingual, style, video_size(media.path))
    candidates = [(kind, suffix, content) for kind, (suffix, content) in files.items()]
    if not candidates:
        return []
    candidates.append(("json", "subtitles.json", json.dumps(metadata, ensure_ascii=False, indent=2)))
    # A unique family name avoids replacing pre-existing subtitles or a previous run.
    if any((directory / f"{stem}.{suffix}").exists() for _, suffix, _ in candidates):
        stem += "." + stamp
    saved = []
    for kind, suffix, content in candidates:
        path = directory / f"{stem}.{suffix}"
        with path.open("x", encoding="utf-8", newline="\n") as output:
            output.write(content)
        saved.append({"kind": kind, "path": str(path)})
    return saved


def key_for(keys, provider):
    key = keys if isinstance(keys, str) else (keys or {}).get(provider, "")
    if not key:
        raise UserError(f"请先在“API Key 设置”中保存{PROVIDERS[provider]['name']}的 Key。")
    return key


class CloudAsr:
    parallel, batch_size, uploads = 4, 1, True

    def __init__(self, client, provider, model, api_key, language):
        self.client, self.provider, self.api_key, self.language = client, provider, api_key, language
        # SiliconFlow keeps its original cache namespace so earlier results are reused.
        self.namespace = "asr:" + model if provider == "siliconflow" else f"asr:{provider}:{model}:{language or 'auto'}"

    def start(self, cancel, progress):
        pass

    def transcribe_batch(self, audios, cancel):
        results = []
        for audio in audios:
            payload, trace = self.client.transcribe(audio, "speech.wav", self.api_key, language=self.language)
            results.append((payload["text"], trace))
        return results

    def close(self):
        pass


class LocalAsr:
    parallel, uploads = 1, False

    def __init__(self, engine, folder, language):
        self.engine, self.language = engine, language
        self.namespace = f"asr:local:{folder}:{language or 'auto'}"

    @property
    def batch_size(self):
        return self.engine.batch_size

    def start(self, cancel, progress):
        self.engine.start(cancel, progress)

    def transcribe_batch(self, audios, cancel):
        return [(text, "") for text in self.engine.transcribe_batch(audios, self.language, cancel)]

    def close(self):
        self.engine.close()


class SubtitlePipeline:
    def __init__(self, clients, cache_dir=None, work_dir=None, detector=None, extractor=None, local_factory=None):
        # A single client object serves every cloud provider (used by tests).
        self.clients = clients if isinstance(clients, dict) else {provider: clients for provider in PROVIDERS}
        self.cache = ResultCache(cache_dir or ROOT / "cache")
        self.work = Path(work_dir or ROOT / "work")
        self.work.mkdir(parents=True, exist_ok=True)
        self.detector = detector if detector is not None else SpeechDetector()
        self.extractor = extractor if extractor is not None else extract_audio
        self.local_factory = local_factory or (lambda folder: LocalAsrEngine(folder, batch_size=16 if "0.6B" in folder else 8))

    def make_asr(self, options, keys):
        engine = ASR_ENGINES[options.get("asr_engine", "siliconflow")]
        source = options.get("source_language", "auto")
        language = None if source == "auto" else source
        if engine["provider"] == "local":
            return LocalAsr(self.local_factory(engine["model"]), engine["model"], language)
        return CloudAsr(self.clients[engine["provider"]], engine["provider"], engine["model"],
                        key_for(keys, engine["provider"]), language)

    def translate_all(self, originals, language, source_language, model, api_key, update, cancel, stats, traces,
                      context_hint="", provider="siliconflow"):
        client = self.clients[provider]
        prefix = f"translation:v2:{model}" if provider == "siliconflow" else f"translation:v2:{provider}:{model}"
        batches = [originals[i:i + TRANSLATION_BATCH] for i in range(0, len(originals), TRANSLATION_BATCH)]
        lock, halt, results = Lock(), Event(), {}

        def translate_batch(number):
            check_cancel(cancel)
            if halt.is_set():
                raise Cancelled("前序翻译失败，已停止后续提交。")
            first, batch = number * TRANSLATION_BATCH, batches[number]
            # Neighbouring lines give the translator context without being translated twice.
            before = [c["text"] for c in originals[max(0, first - 6):first]]
            after = [c["text"] for c in originals[first + len(batch):first + len(batch) + 3]]
            request = {"subtitles": [{"id": c["id"], "text": c["text"]} for c in batch], "before": before,
                       "after": after, "hint": context_hint}
            cache_key = self.cache.key(f"{prefix}:{source_language}:{language}", request)
            cached = self.cache.get(cache_key)
            if isinstance(cached, dict) and all(isinstance(cached.get(str(c["id"])), str) and cached[str(c["id"])].strip() for c in batch):
                with lock:
                    stats["translation_cached"] += 1
                return number, {c["id"]: cached[str(c["id"])] for c in batch}
            check_cancel(cancel)
            if halt.is_set():
                raise Cancelled("已停止后续提交。")
            with lock:
                stats["translation_requests"] += 1
            mapping, usage, trace = client.translate(batch, language, api_key, source_language=source_language,
                                                     model=model, context_before=before, context_after=after,
                                                     context_hint=context_hint)
            self.cache.put(cache_key, {str(k): v for k, v in mapping.items()})
            with lock:
                for field in ("prompt_tokens", "completion_tokens"):
                    if isinstance(usage.get(field), int):
                        stats[field] += usage[field]
                if trace:
                    traces.append(trace)
            return number, mapping

        def guarded(number):
            try:
                return translate_batch(number)
            except Exception:
                halt.set()
                raise

        first_error = None
        update(stage=f"正在翻译为{LANGUAGES[language]}：共 {len(batches)} 批", stats=dict(stats))
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="subtitle-translate") as executor:
            futures = [executor.submit(guarded, number) for number in range(len(batches))]
            for future in as_completed(futures):
                try:
                    number, mapping = future.result()
                    results[number] = mapping
                except CancelledError:
                    continue
                except Exception as exc:
                    if first_error is None or isinstance(first_error, Cancelled) and not isinstance(exc, Cancelled):
                        first_error = exc
                    halt.set()
                    for pending in futures:
                        pending.cancel()
                partial = sorted(({**cue, "text": results[n][cue["id"]], "original_text": cue["text"]}
                                  for n in results for cue in batches[n]), key=lambda c: c["start"])
                with lock:
                    snapshot = dict(stats)
                update(stage=f"字幕翻译：{len(partial)} / {len(originals)} 句", segments=partial, stats=snapshot,
                       progress=round(80 + len(results) / len(batches) * 17, 1))
        check_cancel(cancel)
        if first_error is not None:
            raise first_error
        return [{**cue, "text": results[n][cue["id"]], "original_text": cue["text"]}
                for n in range(len(batches)) for cue in batches[n]]

    def run(self, media, options, keys, update, cancel):
        start, duration = options["clip_start"], options["clip_duration"]
        language = options["target_language"]
        source_language = options.get("source_language", "auto")
        skip_nonverbal = options.get("skip_nonverbal", False) and source_language == "ja"
        translation_model = options.get("translation_model", TRANSLATION_MODEL)
        translation_provider = options.get("translation_provider", "siliconflow")
        translation_key = key_for(keys, translation_provider) if language != "original" else ""
        asr = self.make_asr(options, keys)
        stats = {"asr_requests": 0, "asr_cached": 0, "translation_requests": 0,
                 "translation_cached": 0, "uploaded_seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0,
                 "filtered_segments": 0, "nonverbal_skipped": 0}
        originals, translated, removed = [], [], []
        estimated_wav_bytes = duration * 32000
        if shutil.disk_usage(self.work).free < estimated_wav_bytes + 100 * 1024 * 1024:
            raise UserError("本机剩余磁盘不足以提取这段音频，请缩短处理范围或释放空间。")
        with tempfile.TemporaryDirectory(prefix="subtitle-", dir=self.work) as temporary:
            wav = Path(temporary) / "audio.wav"
            update(stage="正在本地提取音频", progress=0, stats=dict(stats))
            actual_duration = self.extractor(media, wav, start, duration,
                lambda ratio, label: update(stage=label, progress=round(min(ratio, 1) * 10, 1)), cancel)
            check_cancel(cancel)
            ranges = self.detector.detect(wav,
                lambda ratio, label: update(stage=label, progress=round(10 + min(ratio, 1) * 15, 1)), cancel)
            update(total_segments=len(ranges), duration_seconds=actual_duration, stage=f"准备识别 {len(ranges)} 句语音", progress=25)
            if not ranges:
                update(stage="未检测到语音，没有调用付费识别接口", progress=100, stats=dict(stats))
                return "empty"
            halt, stats_lock = Event(), Lock()
            first_error = None
            traces = []
            size = max(1, asr.batch_size)
            batches = [list(range(i, min(i + size, len(ranges)))) for i in range(0, len(ranges), size)]

            def recognize_batch(indices):
                check_cancel(cancel)
                if halt.is_set():
                    raise Cancelled("前序片段失败，已停止后续提交。")
                results, pending = {}, []
                for index in indices:
                    audio = read_wav_segment(wav, ranges[index]["start"], ranges[index]["end"])
                    cache_key = self.cache.key(asr.namespace, audio)
                    cached = self.cache.get(cache_key)
                    if isinstance(cached, dict) and isinstance(cached.get("text"), str):
                        with stats_lock:
                            stats["asr_cached"] += 1
                        results[index] = (cached["text"], "")
                    else:
                        pending.append((index, audio, cache_key))
                if pending:
                    check_cancel(cancel)
                    if halt.is_set():
                        raise Cancelled("已停止后续提交。")
                    with stats_lock:
                        stats["asr_requests"] += len(pending)
                        if asr.uploads:
                            stats["uploaded_seconds"] += sum(ranges[i]["end"] - ranges[i]["start"] for i, _, _ in pending)
                    outputs = asr.transcribe_batch([audio for _, audio, _ in pending], cancel)
                    for (index, _, cache_key), (text, trace) in zip(pending, outputs):
                        self.cache.put(cache_key, {"text": text})
                        results[index] = (text, trace)
                return [({"id": i + 1, "start": start + ranges[i]["start"], "end": start + ranges[i]["end"],
                          "text": clean_text(results[i][0])}, results[i][1]) for i in indices]

            def recognize(indices):
                try:
                    return recognize_batch(indices)
                except Exception:
                    halt.set()
                    raise

            completed = 0
            try:
                asr.start(cancel, lambda label: update(stage=label))
                with ThreadPoolExecutor(max_workers=asr.parallel, thread_name_prefix="subtitle-asr") as executor:
                    futures = [executor.submit(recognize, batch) for batch in batches]
                    for future in as_completed(futures):
                        try:
                            for cue, trace in future.result():
                                if cue["text"] and not language_matches(cue["text"], source_language):
                                    # Very short clips are sometimes recognised as another language.
                                    stats["filtered_segments"] += 1
                                    removed.append({**cue, "reason": "识别语言与视频语言不符"})
                                elif cue["text"] and skip_nonverbal and is_nonverbal(cue["text"]):
                                    stats["nonverbal_skipped"] += 1
                                    removed.append({**cue, "reason": "纯呻吟 / 喘息等语气声"})
                                elif cue["text"]:
                                    originals.append(cue)
                                    originals.sort(key=lambda c: c["start"])
                                if trace:
                                    traces.append(trace)
                                completed += 1
                        except CancelledError:
                            continue
                        except Exception as exc:
                            if first_error is None or isinstance(first_error, Cancelled) and not isinstance(exc, Cancelled):
                                first_error = exc
                            halt.set()
                            for pending in futures:
                                pending.cancel()
                        current = list(originals) if language == "original" else []
                        update(stage=f"语音识别：{completed} / {len(ranges)} 句", progress=round(25 + completed / len(ranges) * 55, 1),
                               processed_segments=completed, cached_segments=stats["asr_cached"], stats=dict(stats),
                               original_segments=list(originals), segments=current)
            finally:
                asr.close()
            check_cancel(cancel)
            if first_error is not None:
                raise first_error
            if not originals:
                update(stage="语音识别未返回文字，没有生成空字幕", progress=100)
                return "empty"
            if language == "original":
                translated = [dict(cue) for cue in originals]
            else:
                translated = self.translate_all(originals, language, source_language, translation_model,
                                                translation_key, update, cancel, stats, traces,
                                                options.get("context_hint", ""), translation_provider)
            check_cancel(cancel)
            update(stage="正在保存 SRT 字幕", progress=98, segments=translated, original_segments=originals)
            engine = ASR_ENGINES[options.get("asr_engine", "siliconflow")]
            metadata = {"model": engine["model"], "asr_engine": options.get("asr_engine", "siliconflow"),
                        "translation_model": translation_model if language != "original" else None,
                        "translation_provider": translation_provider if language != "original" else None,
                        "source": str(media.path), "source_language": source_language,
                        "target_language": language, "time_axis": TIME_AXIS_NOTE, "removed_segments": removed,
                        "clip_start": start, "duration_seconds": actual_duration, "original_segments": originals,
                        "segments": translated, "stats": stats, "trace_ids": traces, "partial": False,
                        "style": options.get("style")}
            try:
                saved = export_files(media, language, originals, translated, options["bilingual"], options["output_dir"],
                                     metadata, style=options.get("style"))
            except (OSError, UserError):
                update(stage="字幕已生成，自动保存失败，请手动下载", progress=100, stats=dict(stats),
                       warning=TIME_AXIS_NOTE + " 自动保存失败，请检查文件夹写入权限；可在客户端手动下载结果。")
            else:
                update(stage="字幕生成完成，已保存到本机", progress=100, saved_files=saved, stats=dict(stats))
            return "done"
