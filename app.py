from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from threading import Event, Lock, Thread
import argparse
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import time

from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from common import (ASR_ENGINES, Cancelled, DEFAULT_ASR, DEFAULT_TRANSLATION, LANGUAGES, PROVIDERS, SOURCE_LANGUAGES,
                    TRANSLATION_MODELS, UserError, safe_message)
from demosaic import SAMPLE_SECONDS, DemosaicStore
from local_asr import gpu_memory, local_status
from media import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, MediaLibrary, choose_path, clip_range, video_size
from providers import API_FORMATS, CustomClient, DeepInfraClient, SiliconFlowClient, custom_endpoint, endpoint_origin
from subtitles import (DEFAULT_STYLE, ROOT, SUBTITLE_FONTS, TIME_AXIS_NOTE, SubtitlePipeline, export_files, file_suffix,
                       make_srt, normalize_style, rewrite_files, unused_path)

ACCOUNT = "api-key"
CUSTOM_SERVICE = "ASR-Web-Custom"
CUSTOM_LABEL = "自定义 API"
# One line sent by "测试连接" to check the address, the key, the model and the answer format in one go.
CUSTOM_TEST_CUE = {"id": 1, "text": "こんにちは、今日はいい天気ですね。"}
MAX_AUDIO_BYTES = 50 * 1024 * 1024
MAX_EDITED_CUES = 20000
EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


class CredentialStore:
    def __init__(self, service=PROVIDERS["siliconflow"]["service"]):
        from keyring.backends.Windows import WinVaultKeyring
        self.service = service
        self.backend = WinVaultKeyring()

    def get(self):
        return self.backend.get_password(self.service, ACCOUNT) or ""

    def set(self, value):
        self.backend.set_password(self.service, ACCOUNT, value)

    def clear(self):
        if self.get():
            self.backend.delete_password(self.service, ACCOUNT)


class MemoryCredentials:
    """Keeps a key for the life of the process only (isolated instances and tests never touch the Windows vault)."""

    def __init__(self):
        self.value = ""

    def get(self):
        return self.value

    def set(self, value):
        self.value = value

    def clear(self):
        self.value = ""


class Settings:
    """settings.json: the subtitle style, the options of the last job and the custom translation API (never its key)."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock = Lock()

    def read(self):
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
        return data if isinstance(data, dict) else {}

    def write(self, key, value):
        with self.lock:
            data = self.read()
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(self.path)

    def style(self):
        # One style is shared by the preview and every .ass export.
        return normalize_style(self.read().get("subtitle_style"), strict=False)

    def preferences(self):
        value = self.read().get("preferences")
        return value if isinstance(value, dict) else {}

    def custom(self):
        """The saved custom translation API, or None when there is none (or the file was edited into something invalid)."""
        value = self.read().get("custom_translation")
        try:
            # Settings saved before there was a choice of format are Chat Completions.
            api_format = value.get("api_format", "chat")
            return {"base_url": value["base_url"], "model": custom_model(value["model"]), "api_format": api_format,
                    "endpoint": custom_endpoint(value["base_url"], api_format)}
        except (TypeError, KeyError, AttributeError, UserError):
            return None


def custom_model(value):
    text = value.strip() if isinstance(value, str) else ""
    if not text or len(text) > 200 or re.search(r"[\x00-\x1f\x7f]", text):
        raise UserError("请填写模型名称，例如 deepseek-chat。")
    return text


class NotFound(UserError):
    pass


def redact(value, keys):
    text = str(value)
    for key in (keys.values() if isinstance(keys, dict) else [keys]):
        if key:
            text = text.replace(key, "[API Key 已隐藏]")
    return safe_message(text)


def timestamped_segments(payload):
    segments = []
    for item in payload.get("segments", []) if isinstance(payload.get("segments"), list) else []:
        if not isinstance(item, dict):
            return []
        start, end, text = item.get("start"), item.get("end"), item.get("text")
        if isinstance(start, bool) or isinstance(end, bool) or not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return []
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end <= start or not isinstance(text, str) or not text.strip():
            return []
        if segments and start < segments[-1]["end"]:
            return []
        segments.append({"start": start, "end": end, "text": text.strip()})
    return segments


@dataclass
class Job:
    id: str
    filename: str
    started_at: str
    revision: int = 0
    status: str = "running"
    stage: str = "正在准备任务"
    progress: float = 0
    seconds: float = 0
    error: str = ""
    warning: str = TIME_AXIS_NOTE
    target_language: str = "zh-CN"
    bilingual: bool = True
    source_language: str = "ja"
    asr_engine: str = DEFAULT_ASR
    translation_quality: str = DEFAULT_TRANSLATION
    translation_model: str = ""
    skip_nonverbal: bool = True
    context_hint: str = ""
    source_path: str = ""
    output_directory: str = ""
    duration_seconds: float = 0
    processed_segments: int = 0
    total_segments: int = 0
    cached_segments: int = 0
    segments: list = field(default_factory=list)
    original_segments: list = field(default_factory=list)
    saved_files: list = field(default_factory=list)
    edited: bool = False
    generated_segments: list = field(default_factory=list)
    generated_original_segments: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    cancel: Event = field(default_factory=Event, repr=False)
    clock: float = field(default_factory=time.monotonic, repr=False)

    def translation_name(self):
        if self.target_language == "original":
            return ""
        if self.translation_quality == "custom":
            return f"自定义 API · {self.translation_model or '未知模型'}"
        return TRANSLATION_MODELS.get(self.translation_quality, {}).get("label", "—")

    def public(self, full=True):
        """The job as the page sees it; full=False leaves out the subtitles themselves (progress polls of a running job)."""
        elapsed = time.monotonic() - self.clock if self.status == "running" else self.seconds
        engine = ASR_ENGINES.get(self.asr_engine, {})
        content = {
            "text": "\n".join(s["text"] for s in self.segments),
            "cues": [{"start": s["start"], "end": s["end"], "text": s["text"], "original_text": s.get("original_text") or ""}
                     for s in self.segments],
        } if full else {}
        return {
            **content,
            "id": self.id, "filename": self.filename, "status": self.status, "revision": self.revision,
            "started_at": self.started_at, "elapsed_seconds": round(elapsed, 1),
            "stage": self.stage, "progress": self.progress, "error": self.error, "warning": self.warning,
            "target_language": self.target_language, "target_language_name": LANGUAGES.get(self.target_language, self.target_language),
            "bilingual": self.bilingual, "source_path": self.source_path, "output_directory": self.output_directory,
            "source_language": self.source_language, "source_language_name": SOURCE_LANGUAGES.get(self.source_language, self.source_language),
            "asr_engine": self.asr_engine, "asr_engine_name": engine.get("label", self.asr_engine),
            "local_asr": engine.get("provider") == "local",
            "translation_quality": self.translation_quality, "skip_nonverbal": self.skip_nonverbal,
            "context_hint": self.context_hint, "translation_model_name": self.translation_name(),
            "duration_seconds": self.duration_seconds, "processed_segments": self.processed_segments,
            "total_segments": self.total_segments, "cached_segments": self.cached_segments,
            "stats": dict(self.stats), "saved_files": list(self.saved_files), "subtitle_count": len(self.segments),
            "edited": self.edited, "can_restore": bool(self.generated_segments),
            "has_timestamps": bool(self.segments), "has_original": bool(self.original_segments),
            "has_bilingual": bool(self.segments) and self.bilingual and self.target_language != "original",
            "cancel_requested": self.cancel.is_set(), "partial": self.status not in {"done", "empty"},
        }


class JobStore:
    def __init__(self, pipeline, history_dir=None):
        self.pipeline = pipeline
        self.lock = Lock()
        self.jobs = OrderedDict()
        self.revision = 0
        self.history = Path(history_dir or ROOT / "history")
        self.history.mkdir(parents=True, exist_ok=True)
        # Called with a job that finished successfully (outside the lock).
        self.on_done = None
        self.restore_latest()

    def restore_latest(self):
        files = sorted(self.history.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in files[:5]:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if data.get("status") not in {"done", "failed", "cancelled", "empty"}:
                    continue
                fields = {name for name in Job.__dataclass_fields__ if name not in {"cancel", "clock"}}
                job = Job(**{k: v for k, v in data.items() if k in fields})
                self.jobs[job.id] = job
                self.revision = job.revision
                break
            except (OSError, ValueError, TypeError):
                continue

    def persist(self, job):
        values = {name: getattr(job, name) for name in Job.__dataclass_fields__ if name not in {"cancel", "clock"}}
        try:
            path = self.history / f"{job.id}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(values, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)
        except OSError:
            pass

    def start(self, media, options, keys, expected_revision, cleanup=None):
        with self.lock:
            if expected_revision != self.revision:
                raise UserError("此页面的任务状态已过期，可能已在另一个窗口提交。请刷新查看最新结果；重新生成会复用成功片段缓存，缺失部分仍可能计费。")
            if any(job.status == "running" for job in self.jobs.values()):
                raise UserError("已有任务正在处理，请等待或取消，避免重复扣费。")
            self.revision += 1
            job = Job(secrets.token_hex(16), media.path.name, datetime.now(timezone.utc).isoformat(), revision=self.revision,
                      target_language=options["target_language"], bilingual=options["bilingual"],
                      source_language=options["source_language"], asr_engine=options["asr_engine"],
                      translation_quality=options["translation_quality"],
                      translation_model=options.get("translation_model", ""),
                      skip_nonverbal=options["skip_nonverbal"], context_hint=options["context_hint"],
                      source_path=str(media.path), output_directory=options["output_dir"], duration_seconds=options["clip_duration"])
            self.jobs[job.id] = job
            while len(self.jobs) > 10:
                self.jobs.popitem(last=False)
        Thread(target=self.run, args=(job, media, options, keys, cleanup), daemon=True).start()
        return job

    def run(self, job, media, options, keys, cleanup):
        def update(**changes):
            with self.lock:
                for key, value in changes.items():
                    if key in Job.__dataclass_fields__ and key not in {"id", "cancel", "clock", "status"}:
                        setattr(job, key, value)
        status = "failed"
        try:
            status = self.pipeline.run(media, options, keys, update, job.cancel)
        except Cancelled as exc:
            status = "cancelled"
            job.error = redact(exc, keys)
            job.stage = "已取消"
        except UserError as exc:
            job.error = redact(exc, keys)
            job.stage = "处理失败，已完成的语音片段可复用缓存"
        except Exception:
            job.error = "本机处理发生异常。请查看磁盘空间、文件权限和网络；成功的语音片段已缓存，不会自动重复请求。"
            job.stage = "处理失败"
        if status in {"failed", "cancelled"} and (job.segments or job.original_segments):
            try:
                metadata = {"partial": True, "error": job.error, "model": ASR_ENGINES.get(job.asr_engine, {}).get("model"),
                            "asr_engine": job.asr_engine, "target_language": job.target_language,
                            "time_axis": TIME_AXIS_NOTE, "original_segments": job.original_segments,
                            "segments": job.segments, "stats": job.stats, "style": options.get("style")}
                job.saved_files = export_files(media, job.target_language, job.original_segments, job.segments,
                                              job.bilingual, job.output_directory, metadata, partial=True,
                                              style=options.get("style"))
            except (OSError, UserError):
                job.warning += " 自动保存部分结果失败，仍可在客户端手动下载。"
        if cleanup is not None:
            try:
                cleanup.unlink(missing_ok=True)
            except OSError:
                pass
        with self.lock:
            job.seconds = time.monotonic() - job.clock
            job.status = status
            self.persist(job)
        if status == "done" and self.on_done is not None:
            try:
                self.on_done(job)
            except (OSError, UserError):
                with self.lock:
                    job.warning += " 没能为去码视频复制字幕文件。"

    def get(self, ident):
        with self.lock:
            return self.jobs.get(ident)

    def latest(self):
        with self.lock:
            return next(reversed(self.jobs.values()), None)


def tidy_text(value):
    lines = [" ".join(re.sub(r"[\x00-\x1f\x7f]", " ", line).split()) for line in value.replace("\r", "").split("\n")]
    return "\n".join(line for line in lines if line)


def same_cues(current, edited):
    """True when the editor sent the subtitles back unchanged (for example only the style changed)."""
    if len(current) != len(edited):
        return False
    for old, new in zip(current, edited):
        if abs(old["start"] - new["start"]) > 0.0015 or abs(old["end"] - new["end"]) > 0.0015:
            return False
        if any(tidy_text(old.get(key) or "") != (new.get(key) or "") for key in ("text", "original_text")):
            return False
    return True


def editable_text(value, label, index, required=True):
    if value is None and not required:
        return ""
    if not isinstance(value, str):
        raise UserError(f"第 {index} 条字幕的{label}格式不正确。")
    text = tidy_text(value)
    if required and not text:
        raise UserError(f"第 {index} 条字幕的{label}为空，请填写或删除这条字幕。")
    if len(text) > 1000:
        raise UserError(f"第 {index} 条字幕太长，请控制在 1000 字以内。")
    return text


def parse_edited_cues(cues, translated):
    if not isinstance(cues, list) or not cues:
        raise UserError("至少需要保留一条字幕。")
    if len(cues) > MAX_EDITED_CUES:
        raise UserError("字幕条数过多。")
    parsed = []
    for index, cue in enumerate(cues, 1):
        if not isinstance(cue, dict):
            raise UserError(f"第 {index} 条字幕格式不正确。")
        start, end = cue.get("start"), cue.get("end")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in (start, end)):
            raise UserError(f"第 {index} 条字幕的时间无效。")
        if start < 0 or end > 24 * 3600 or end - start < 0.05:
            raise UserError(f"第 {index} 条字幕的结束时间必须晚于开始时间。")
        if parsed and start < parsed[-1]["end"] - 0.002:
            raise UserError(f"第 {index} 条字幕与上一条时间重叠，请调整起止时间。")
        item = {"id": index, "start": round(float(start), 3), "end": round(float(end), 3),
                "text": editable_text(cue.get("text"), "字幕文字", index)}
        if translated:
            item["original_text"] = editable_text(cue.get("original_text"), "原文", index, required=False)
        parsed.append(item)
    return parsed


def validated_output(path):
    directory = Path(str(path).strip().strip('"'))
    if str(directory).startswith(("\\\\", "//")) or not directory.is_absolute() or not directory.is_dir():
        raise UserError("请选择本机已经存在的字幕保存文件夹。")
    return str(directory.resolve())


def create_app(credential_store=None, client=None, sample_path=None, media_library=None, pipeline=None, picker=None,
               history_dir=None, local_check=None, settings_path=None, demosaic=None, custom_client=CustomClient):
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_AUDIO_BYTES + 1024 * 1024
    # The key of the custom translation API has its own store: a built-in provider's key never goes to another server.
    if credential_store is None:
        credentials = {provider: CredentialStore(info["service"]) for provider, info in PROVIDERS.items()}
        custom_key = CredentialStore(CUSTOM_SERVICE)
    elif isinstance(credential_store, dict):
        credentials = {provider: store for provider, store in credential_store.items() if provider in PROVIDERS}
        custom_key = credential_store.get("custom") or MemoryCredentials()
    else:
        credentials = {provider: credential_store for provider in PROVIDERS}
        custom_key = MemoryCredentials()
    library = media_library if media_library is not None else MediaLibrary()
    clients = client if client is not None else {"siliconflow": SiliconFlowClient(), "deepinfra": DeepInfraClient()}
    jobs = JobStore(pipeline if pipeline is not None else SubtitlePipeline(clients, custom_factory=custom_client), history_dir)
    # Settings live next to the history folder, so a test or isolated instance never touches the real preference.
    settings = Settings(settings_path or jobs.history.parent / "settings.json")
    picker = picker if picker is not None else choose_path
    local_check = local_check if local_check is not None else local_status
    # The sample is only used by automated tests; the desktop client always picks local files.
    sample = Path(sample_path) if sample_path is not None else None
    sample_media = None
    if sample is not None and sample.is_file():
        try:
            sample_media = library.add(sample, "sample")
        except UserError:
            pass
    restorer = demosaic if demosaic is not None else DemosaicStore()
    csrf_token = secrets.token_urlsafe(32)
    state = {"last_media": None}
    app.extensions["asr_jobs"] = jobs
    app.extensions["asr_media"] = library
    app.extensions["asr_demosaic"] = restorer

    def link_subtitles(job, restored):
        """Copy a job's subtitles next to a restored video under its name, so players load them with it.

        The copies join the job's files, so saving in the editor updates them as well."""
        restored = Path(restored)
        with jobs.lock:
            if job.status != "done" or any(item.get("video") == str(restored) for item in job.saved_files):
                return []
            copies = []
            for item in job.saved_files:
                path = Path(item["path"])
                if item["kind"] == "json" or item.get("video") or not path.is_file():
                    continue
                target = unused_path(restored.with_name(f"{restored.stem}.{file_suffix(item['kind'], job.target_language)}"))
                shutil.copyfile(path, target)
                copies.append({"kind": item["kind"], "path": str(target), "video": str(restored)})
            if copies:
                job.saved_files = job.saved_files + copies
                jobs.persist(job)
        return [item["path"] for item in copies]

    def restored_video_done(source_path, restored):
        with jobs.lock:
            job = next((j for j in reversed(jobs.jobs.values()) if j.source_path == source_path and j.status == "done"), None)
        return link_subtitles(job, restored) if job is not None else []

    def subtitles_done(job):
        restored = restorer.latest_output(job.source_path) if job.source_path else None
        if restored:
            link_subtitles(job, restored)

    restorer.on_done = restored_video_done
    jobs.on_done = subtitles_done

    def key_status():
        status = {}
        for provider, store in credentials.items():
            try:
                status[provider] = bool(store.get())
            except Exception:
                status[provider] = False
        return status

    def provider_arg():
        data = request.get_json(silent=True)
        provider = data.get("provider", "siliconflow") if isinstance(data, dict) else "siliconflow"
        if provider not in PROVIDERS:
            raise UserError("未知的服务商。")
        return provider, data if isinstance(data, dict) else {}

    def custom_public():
        saved = settings.custom()
        try:
            has_key = bool(custom_key.get())
        except Exception:
            has_key = False
        return {"configured": saved is not None, "key_saved": has_key,
                **(saved or {"base_url": "", "model": "", "api_format": "chat", "endpoint": ""})}

    def translation_choices():
        custom = settings.custom()
        return [*({"code": code, "name": item["label"], "model": item["model"], "provider": item["provider"]}
                  for code, item in TRANSLATION_MODELS.items()),
                {"code": "custom", "name": CUSTOM_LABEL + (f" · {custom['model']}" if custom else ""),
                 "model": custom["model"] if custom else "", "provider": "custom"}]

    def custom_request():
        """The custom API a request describes: (its settings, the key typed with it, whether a saved key belongs to it)."""
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        api_format = data.get("api_format", "chat")
        endpoint = custom_endpoint(data.get("base_url"), api_format)
        model = custom_model(data.get("model"))
        typed = data.get("api_key") or ""
        if not isinstance(typed, str) or typed.strip() and not re.fullmatch(r"[\x21-\x7e]{1,500}", typed.strip()):
            raise UserError("API Key 只能包含英文字母、数字和符号，不能有空格或中文。")
        saved = settings.custom()
        # A saved key stays with the server it was entered for.
        same_server = saved is not None and endpoint_origin(saved["endpoint"]) == endpoint_origin(endpoint)
        values = {"base_url": data["base_url"].strip(), "model": model, "api_format": api_format, "endpoint": endpoint}
        return values, typed.strip(), same_server

    @app.before_request
    def local_only():
        if request.remote_addr not in {"127.0.0.1", "::1"} or request.host.split(":", 1)[0].lower() not in {"127.0.0.1", "localhost"}:
            return jsonify(error="仅允许通过本机地址访问。"), 403
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            if not secrets.compare_digest(request.headers.get("X-ASR-Token", ""), csrf_token):
                return jsonify(error="页面校验失败，请重新打开客户端或刷新页面。"), 403
            origin = request.headers.get("Origin")
            if origin and origin != "http://" + request.host:
                return jsonify(error="不允许跨站请求。"), 403

    @app.after_request
    def secure_headers(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; media-src 'self' blob:; img-src 'self' data:; base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
        return response

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(error):
        return jsonify(error="上传音频不能超过 50MB；大视频请使用“选择本机视频”直接读取，不要上传。"), 413

    @app.errorhandler(UserError)
    def user_error(error):
        return jsonify(error=safe_message(error)), 404 if isinstance(error, NotFound) else 400

    @app.get("/")
    def index():
        return render_template("index.html", csrf_token=csrf_token)

    @app.get("/favicon.ico")
    def favicon():
        return "", 204

    @app.get("/api/health")
    def health():
        return jsonify(service="asr-web-siliconflow", version=3)

    @app.get("/api/config")
    def config():
        latest = jobs.latest()
        keys = key_status()
        engines = []
        for code, engine in ASR_ENGINES.items():
            available, reason = (local_check(engine["model"]) if engine["provider"] == "local" else (True, ""))
            engines.append({"code": code, "label": engine["label"], "provider": engine["provider"], "available": available,
                            "reason": reason, "price_per_second": engine["price_per_second"], "vram_mb": engine.get("vram_mb")})
        last = None
        if state["last_media"]:
            try:
                media = library.get(state["last_media"])
                last = media.public() if media.path.is_file() else None
            except UserError:
                last = None
        with jobs.lock:
            recent = latest.public() if latest else None
            revision = jobs.revision
        demosaic_ready, demosaic_reason = restorer.status()
        latest_restore = restorer.latest()
        with restorer.lock:
            recent_restore = latest_restore.public() if latest_restore else None
        return jsonify(demosaic={"available": demosaic_ready, "reason": demosaic_reason, "sample_seconds": SAMPLE_SECONDS},
                       latest_demosaic=recent_restore, api_key_configured=keys.get("siliconflow", False), keys=keys,
                       providers=[{"code": code, "name": info["name"]} for code, info in PROVIDERS.items()],
                       submission_revision=revision, languages=[{"code": code, "name": name} for code, name in LANGUAGES.items()],
                       source_languages=[{"code": code, "name": name} for code, name in SOURCE_LANGUAGES.items()],
                       asr_engines=engines, default_asr=DEFAULT_ASR, gpu=gpu_memory(),
                       translation_models=translation_choices(), custom_translation=custom_public(),
                       custom_formats=[{"code": code, "name": item["name"], "path": item["path"]}
                                       for code, item in API_FORMATS.items()],
                       default_translation=DEFAULT_TRANSLATION, last_media=last, latest_job=recent, desktop_mode=True,
                       preferences=settings.preferences(), subtitle_style=settings.style(), default_style=DEFAULT_STYLE,
                       subtitle_fonts=[{"code": family, "name": label, "cell": cell}
                                       for family, (label, cell) in SUBTITLE_FONTS.items()])

    @app.post("/api/key")
    def save_key():
        provider, data = provider_arg()
        value = data.get("api_key")
        if not isinstance(value, str) or not re.fullmatch(PROVIDERS[provider]["key_pattern"], value.strip()):
            return jsonify(error=f"请输入有效的{PROVIDERS[provider]['name']} API Key。"), 400
        try:
            credentials[provider].set(value.strip())
        except Exception:
            return jsonify(error="无法写入 Windows 凭据管理器，未保存 Key。"), 500
        return jsonify(success=True)

    @app.post("/api/key/clear")
    def clear_key():
        provider, _ = provider_arg()
        try:
            credentials[provider].clear()
        except Exception:
            return jsonify(error="无法清除本机凭据。"), 500
        return jsonify(success=True)

    @app.post("/api/custom-translation")
    def save_custom():
        values, typed, same_server = custom_request()
        try:
            had_key = bool(custom_key.get())
            if had_key and not same_server:
                custom_key.clear()
            settings.write("custom_translation", {key: values[key] for key in ("base_url", "model", "api_format")})
            if typed:
                custom_key.set(typed)
        except Exception:
            return jsonify(error="无法保存自定义接口设置（写入设置文件或 Windows 凭据管理器失败），请重试。"), 500
        return jsonify(custom_translation=custom_public(), translation_models=translation_choices(),
                       key_cleared=had_key and not same_server and not typed)

    @app.post("/api/custom-translation/clear")
    def clear_custom():
        try:
            custom_key.clear()
            settings.write("custom_translation", None)
        except Exception:
            return jsonify(error="无法清除自定义接口设置。"), 500
        return jsonify(custom_translation=custom_public(), translation_models=translation_choices())

    @app.post("/api/custom-translation/test")
    def test_custom():
        values, typed, same_server = custom_request()
        key = typed
        if not key and same_server:
            try:
                key = custom_key.get()
            except Exception:
                return jsonify(error="无法读取本机保存的 API Key。"), 500
        started = time.monotonic()
        try:
            mapping, _, _ = custom_client(values["endpoint"], timeout=60, api_format=values["api_format"]).translate(
                [CUSTOM_TEST_CUE], "zh-CN", key, source_language="ja", model=values["model"])
        except UserError as exc:
            return jsonify(error=redact(exc, key), endpoint=values["endpoint"]), 400
        return jsonify(success=True, source=CUSTOM_TEST_CUE["text"], translation=mapping[CUSTOM_TEST_CUE["id"]],
                       seconds=round(time.monotonic() - started, 1), endpoint=values["endpoint"], used_key=bool(key))

    @app.post("/api/media/pick")
    def pick_media():
        path = picker(directory=False)
        if not path:
            return jsonify(media=None)
        media = library.add(path)
        state["last_media"] = media.id
        return jsonify(media=media.public())

    @app.post("/api/media/load")
    def load_media():
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or not isinstance(data.get("path"), str):
            raise UserError("请输入本机文件的完整路径。")
        media = library.add(data["path"])
        state["last_media"] = media.id
        return jsonify(media=media.public())

    @app.get("/api/media/<ident>/preview")
    def preview(ident):
        media = library.get(ident)
        if not media.path.is_file():
            return jsonify(error="媒体文件已移动或删除。"), 404
        return send_file(media.path, conditional=True)

    @app.get("/api/sample/audio")
    def sample_audio():
        if not sample_media or not sample.is_file():
            return jsonify(error="没有测试样本。"), 404
        return send_file(sample, mimetype="audio/mpeg", conditional=True)

    @app.post("/api/transcribe")
    def transcribe():
        language = request.form.get("target_language", "zh-CN")
        if language not in LANGUAGES:
            raise UserError("请选择受支持的目标语言。")
        source_language = request.form.get("source_language", "ja")
        if source_language not in SOURCE_LANGUAGES:
            raise UserError("请选择受支持的视频语言。")
        asr_engine = request.form.get("asr_engine", DEFAULT_ASR)
        if asr_engine not in ASR_ENGINES:
            raise UserError("请选择受支持的语音识别方式。")
        engine = ASR_ENGINES[asr_engine]
        if engine["provider"] == "local":
            available, reason = local_check(engine["model"])
            if not available:
                raise UserError(f"{reason}，请改用云端识别。")
        quality = request.form.get("translation_quality", DEFAULT_TRANSLATION)
        custom = settings.custom() if quality == "custom" else None
        if quality == "custom":
            if custom is None and language != "original":
                raise UserError("请先在“翻译模型”下方填写并保存自定义 API 的接口地址和模型名称。")
            translation = {"provider": "custom", "model": custom["model"] if custom else ""}
        elif quality not in TRANSLATION_MODELS:
            raise UserError("请选择受支持的翻译模型。")
        else:
            translation = TRANSLATION_MODELS[quality]
        context_hint = " ".join(re.sub(r"[\x00-\x1f\x7f]", " ", request.form.get("context_hint", "")).split())
        if len(context_hint) > 200:
            raise UserError("内容背景最多 200 字。")
        revision = request.form.get("submission_revision", "")
        if not revision.isascii() or not revision.isdigit() or len(revision) > 10:
            raise UserError("页面缺少有效任务状态，请重新打开客户端或刷新后再提交。")
        needed = {engine["provider"]} - {"local"}
        if language != "original" and custom is None:
            needed.add(translation["provider"])
        keys = {}
        for provider in sorted(needed):
            try:
                keys[provider] = credentials[provider].get()
            except Exception:
                return jsonify(error="无法读取本机保存的 API Key。"), 500
            if not keys[provider]:
                raise UserError(f"请先在“API Key 设置”中保存{PROVIDERS[provider]['name']}的 Key。")
        if language != "original" and custom is not None:
            try:
                secret = custom_key.get()
            except Exception:
                return jsonify(error="无法读取本机保存的 API Key。"), 500
            if secret:
                # Optional: a model server on this machine needs no key.
                keys["custom"] = secret
        cleanup = None
        if request.form.get("use_sample") == "true":
            if not sample_media or not sample.is_file():
                return jsonify(error="没有测试样本，请选择视频。"), 404
            media = sample_media
        elif request.form.get("media_id"):
            media = library.get(request.form["media_id"])
        else:
            upload = request.files.get("file")
            if upload is None or not upload.filename:
                raise UserError("请先选择视频或音频。")
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in AUDIO_EXTENSIONS:
                raise UserError("上传模式仅接收音频；大视频请使用本机文件选择。")
            audio = upload.read(MAX_AUDIO_BYTES + 1)
            if not audio:
                raise UserError("音频文件为空。")
            if len(audio) > MAX_AUDIO_BYTES:
                return jsonify(error="音频不能超过 50MB。"), 413
            incoming = ROOT / "uploads"
            incoming.mkdir(exist_ok=True)
            cleanup = incoming / (secrets.token_hex(16) + suffix)
            cleanup.write_bytes(audio)
            try:
                media = library.add(cleanup)
            except Exception:
                cleanup.unlink(missing_ok=True)
                raise
        try:
            start_value, duration_value = request.form.get("clip_start", "0"), request.form.get("clip_duration", "")
            try:
                start, duration = clip_range(media, float(start_value or "0"), float(duration_value) if duration_value else None)
            except ValueError:
                raise UserError("起始时间和处理时长必须为秒数。") from None
            fallback = ROOT / "results" if cleanup is not None else media.path.parent
            fallback.mkdir(exist_ok=True)
            output_dir = validated_output(request.form.get("output_dir") or fallback)
            options = {"target_language": language, "bilingual": request.form.get("bilingual", "true") == "true",
                       "clip_start": start, "clip_duration": duration, "output_dir": output_dir,
                       "source_language": source_language, "asr_engine": asr_engine,
                       "skip_nonverbal": request.form.get("skip_nonverbal", "true") == "true",
                       "translation_quality": quality, "translation_model": translation["model"],
                       "translation_provider": translation["provider"], "context_hint": context_hint,
                       "translation_endpoint": custom["endpoint"] if custom else "",
                       "translation_format": custom["api_format"] if custom else "", "style": settings.style()}
            job = jobs.start(media, options, keys, int(revision), cleanup)
            try:
                # The next visit starts from the options of this job. The two checkboxes are only remembered from
                # a job they applied to (the page switches them off for untranslated or non-Japanese jobs).
                remembered = ["asr_engine", "source_language", "target_language", "translation_quality"]
                remembered += ["bilingual"] if language != "original" else []
                remembered += ["skip_nonverbal"] if source_language == "ja" else []
                settings.write("preferences", {**settings.preferences(), **{key: options[key] for key in remembered}})
            except OSError:
                pass
        except UserError as exc:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)
            code = 409 if "状态已过期" in str(exc) or "已有任务" in str(exc) else 400
            return jsonify(error=str(exc)), code
        with jobs.lock:
            return jsonify(job=job.public()), 202

    @app.get("/api/jobs/<ident>")
    def job_status(ident):
        job = jobs.get(ident)
        if job is None:
            return jsonify(error="任务不存在，请重新打开客户端查看最近任务。"), 404
        with jobs.lock:
            # While it runs the page only shows progress; the subtitles follow with the first answer after the end.
            return jsonify(job=job.public(full=job.status != "running"))

    @app.post("/api/jobs/<ident>/cancel")
    def cancel_job(ident):
        job = jobs.get(ident)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        with jobs.lock:
            if job.status == "running":
                job.cancel.set()
                job.stage = "正在取消，等待已发出的请求返回；不会继续提交新片段"
            return jsonify(job=job.public())

    def save_job_segments(job, segments, original_segments, edited, style):
        translated = job.target_language != "original"
        # Render first so invalid subtitles never replace the files on disk.
        make_srt(segments)
        if translated and original_segments:
            make_srt(original_segments)
        update = {"segments": segments, "original_segments": original_segments, "edited": edited, "style": style}
        if job.saved_files:
            job.saved_files = rewrite_files(job.saved_files, job.target_language, original_segments, segments, job.bilingual,
                                            update, style, video_size(job.source_path) if job.source_path else None)
        else:
            metadata = {"model": ASR_ENGINES.get(job.asr_engine, {}).get("model"), "target_language": job.target_language,
                        "time_axis": TIME_AXIS_NOTE, **update}
            job.saved_files = export_files(SimpleNamespace(path=Path(job.source_path)), job.target_language, original_segments,
                                           segments, job.bilingual, job.output_directory, metadata,
                                           partial=job.status != "done", style=style)
        job.segments, job.original_segments, job.edited = segments, original_segments, edited
        jobs.persist(job)

    @app.get("/api/jobs/<ident>/media")
    def job_media(ident):
        job = jobs.get(ident)
        if job is None or not job.source_path or not Path(job.source_path).is_file():
            return jsonify(error="找不到这个任务的视频文件，可能已移动或删除。"), 404
        return send_file(job.source_path, conditional=True)

    @app.post("/api/jobs/<ident>/segments")
    def edit_segments(ident):
        job = jobs.get(ident)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        translated = job.target_language != "original"
        cues = parse_edited_cues(data.get("cues"), translated)
        style = normalize_style(data["style"]) if "style" in data else settings.style()
        segments = [{k: v for k, v in cue.items() if translated or k != "original_text"} for cue in cues]
        original_segments = ([{"id": c["id"], "start": c["start"], "end": c["end"], "text": c["original_text"]}
                              for c in cues if c.get("original_text")] if translated else [dict(c) for c in segments])
        with jobs.lock:
            if job.status == "running":
                return jsonify(error="任务还在处理中，完成后才能编辑字幕。"), 409
            try:
                settings.write("subtitle_style", style)
            except OSError:
                return jsonify(error="无法保存字幕样式设置，字幕文件未改动。"), 500
            try:
                if same_cues(job.segments, segments):
                    # Only the style changed: refresh the files without marking the subtitles as edited.
                    save_job_segments(job, job.segments, job.original_segments, job.edited, style)
                else:
                    if not job.generated_segments:
                        job.generated_segments = list(job.segments)
                        job.generated_original_segments = list(job.original_segments)
                    save_job_segments(job, segments, original_segments, True, style)
            except OSError:
                return jsonify(error="无法写入字幕文件，请检查文件是否被其他程序占用。"), 500
            return jsonify(job=job.public(), subtitle_style=style)

    @app.post("/api/jobs/<ident>/segments/restore")
    def restore_segments(ident):
        job = jobs.get(ident)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        with jobs.lock:
            if job.status == "running":
                return jsonify(error="任务还在处理中。"), 409
            if not job.generated_segments:
                return jsonify(error="这个任务没有编辑过，无需恢复。"), 409
            style = settings.style()
            try:
                save_job_segments(job, list(job.generated_segments), list(job.generated_original_segments), False, style)
            except OSError:
                return jsonify(error="无法写入字幕文件，请检查文件是否被其他程序占用。"), 500
            job.generated_segments, job.generated_original_segments = [], []
            jobs.persist(job)
            return jsonify(job=job.public(), subtitle_style=style)

    def demosaic_job(ident):
        job = restorer.get(ident)
        if job is None:
            raise NotFound("去马赛克任务不存在，请刷新页面。")
        return job

    @app.post("/api/demosaic")
    def start_demosaic():
        data = request.get_json(silent=True)
        data = data if isinstance(data, dict) else {}
        available, reason = restorer.status()
        if not available:
            raise UserError(reason + "。")
        if not isinstance(data.get("media_id"), str):
            raise UserError("请先在第 1 步选择视频。")
        media = library.get(data["media_id"])
        if media.kind != "video":
            raise UserError("去马赛克只能处理视频文件。")
        if not media.path.is_file():
            raise UserError("视频文件已移动或删除，请重新选择。")
        sample = data.get("sample_start")
        if sample is not None:
            if (isinstance(sample, bool) or not isinstance(sample, (int, float)) or not math.isfinite(sample)
                    or not 0 <= sample < media.duration - 0.5):
                raise UserError("试处理的起点必须在视频时长范围内。")
            sample = float(sample)
        try:
            job = restorer.start(media, sample)
        except UserError as exc:
            return jsonify(error=str(exc)), 409
        with restorer.lock:
            return jsonify(job=job.public()), 202

    @app.get("/api/demosaic/<ident>")
    def demosaic_status(ident):
        job = demosaic_job(ident)
        with restorer.lock:
            return jsonify(job=job.public())

    @app.post("/api/demosaic/<ident>/cancel")
    def cancel_demosaic(ident):
        job = demosaic_job(ident)
        with restorer.lock:
            if job.status == "running":
                job.cancel.set()
                job.stage = "正在取消去马赛克…"
            return jsonify(job=job.public())

    @app.get("/api/demosaic/<ident>/media")
    def demosaic_media(ident):
        job = demosaic_job(ident)
        if job.status != "done" or not Path(job.output_path).is_file():
            return jsonify(error="找不到去码后的视频，可能已移动或删除。"), 404
        return send_file(job.output_path, conditional=True)

    @app.post("/api/demosaic/<ident>/reveal")
    def reveal_demosaic(ident):
        job = demosaic_job(ident)
        if job.status != "done" or not Path(job.output_path).is_file():
            return jsonify(error="找不到去码后的视频，可能已移动或删除。"), 404
        # Opens Explorer with the file selected.
        subprocess.Popen(f'explorer /select,"{job.output_path}"')
        return jsonify(success=True)

    @app.post("/api/jobs/<ident>/open-output")
    def open_output(ident):
        job = jobs.get(ident)
        if job is None or not Path(job.output_directory).is_dir():
            return jsonify(error="结果文件夹不存在。"), 404
        os.startfile(job.output_directory)
        return jsonify(success=True)

    @app.get("/api/jobs/<ident>/download/<kind>")
    def download(ident, kind):
        job = jobs.get(ident)
        if job is None:
            return jsonify(error="任务不存在。"), 404
        with jobs.lock:
            if kind == "txt" and job.segments:
                content, mime = "\n".join(s["text"] for s in job.segments), "text/plain; charset=utf-8"
            elif kind == "json" and (job.segments or job.original_segments):
                content = json.dumps({"model": ASR_ENGINES.get(job.asr_engine, {}).get("model"), "asr_engine": job.asr_engine,
                                      "translation_model": job.translation_model or TRANSLATION_MODELS.get(job.translation_quality, {}).get("model"),
                                      "source_language": job.source_language, "target_language": job.target_language,
                                      "partial": job.status != "done", "time_axis": TIME_AXIS_NOTE,
                                      "original_segments": job.original_segments, "segments": job.segments, "stats": job.stats}, ensure_ascii=False, indent=2)
                mime = "application/json; charset=utf-8"
            elif kind == "srt" and job.segments:
                content, mime = make_srt(job.segments), "application/x-subrip; charset=utf-8"
            elif kind == "original" and job.original_segments:
                content, mime = make_srt(job.original_segments), "application/x-subrip; charset=utf-8"
            elif kind == "bilingual" and job.segments and job.bilingual and job.target_language != "original":
                content, mime = make_srt(job.segments, bilingual=True), "application/x-subrip; charset=utf-8"
            else:
                return jsonify(error="该格式的结果尚未生成，可先等待处理完成或下载已有原文。"), 409
            extension = "srt" if kind in {"srt", "original", "bilingual"} else kind
            partial = ".partial" if job.status != "done" else ""
            language = "source" if kind == "original" else job.target_language
            name = f"{Path(job.filename).stem}{partial}.{language}.{kind}.{extension}" if kind == "bilingual" else f"{Path(job.filename).stem}{partial}.{language}.{extension}"
        return send_file(BytesIO(content.encode("utf-8")), mimetype=mime, as_attachment=True, download_name=name)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    create_app().run(host="127.0.0.1", port=args.port, debug=False, threaded=True, use_reloader=False)
