from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import json
import re
import time
import unittest
from unittest.mock import patch

from app import SiliconFlowClient, UserError, create_app, make_srt, safe_message, timestamped_segments
from common import MODEL, Cancelled, check_cancel
from media import Media
from providers import DEEPINFRA_CHAT, DeepInfraClient
from subtitles import DEFAULT_STYLE, export_files

TEST_KEY = "sk-" + "testcredential" * 3
DEEPINFRA_KEY = "di" + "deepinfrakey" * 3


class MemoryCredentials:
    def __init__(self, value=TEST_KEY): self.value = value
    def get(self): return self.value
    def set(self, value): self.value = value
    def clear(self): self.value = ""


class FakeClient:
    def __init__(self, payload=None):
        self.payload = payload if payload is not None else {"text": "こんにちは。"}
        self.calls = []
        self.hold = None
        self.failure = None
        self.translation_calls = []

    def transcribe(self, audio, filename, api_key, language=None):
        self.calls.append((audio, filename, api_key))
        if self.hold: self.hold.wait(5)
        if self.failure: raise self.failure
        return self.payload, "test-trace-id"

    def translate(self, cues, language, api_key):
        self.translation_calls.append((cues, language))
        return {c["id"]: "你好。" for c in cues}, {"prompt_tokens": 20, "completion_tokens": 8}, "translation-trace"


class FakeMediaLibrary:
    def __init__(self): self.items = {}
    def add(self, path, ident=None):
        path = Path(path)
        if not path.is_file(): raise UserError("文件不存在")
        info = path.stat()
        media = Media(ident or str(len(self.items)), path, 10, info.st_size, info.st_mtime_ns, "video" if path.suffix == ".mkv" else "audio")
        self.items[media.id] = media
        return media
    def get(self, ident):
        if ident not in self.items: raise UserError("媒体不存在")
        return self.items[ident]


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.runs = []
    def run(self, media, options, keys, update, cancel):
        self.runs.append((options, dict(keys)))
        key = keys.get("siliconflow", "")
        update(stage="正在提取音频", progress=10)
        check_cancel(cancel)
        payload, trace = self.client.transcribe(media.path.read_bytes(), media.path.name, key)
        check_cancel(cancel)
        original = timestamped_segments(payload) or [{"start": 0.5, "end": 2, "text": payload["text"]}]
        original = [{**cue, "id": index} for index, cue in enumerate(original, 1)]
        update(stage="语音识别完成", progress=80, original_segments=original, processed_segments=len(original), total_segments=len(original))
        if options["target_language"] == "original":
            translated = original
        else:
            mapped, _, _ = self.client.translate(original, options["target_language"], key)
            translated = [{**cue, "text": mapped[cue["id"]], "original_text": cue["text"]} for cue in original]
        saved = export_files(media, options["target_language"], original, translated, options["bilingual"], options["output_dir"], {"response": payload})
        update(stage="已保存字幕", progress=100, segments=translated, saved_files=saved)
        return "done"


class AppTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.sample = Path(self.temp.name) / "sample.mp3"
        self.sample.write_bytes(b"sample-audio")
        self.credentials = {"siliconflow": MemoryCredentials(), "deepinfra": MemoryCredentials("")}
        self.upstream = FakeClient()
        self.library = FakeMediaLibrary()
        self.history = Path(self.temp.name) / "history"
        self.exports = Path(self.temp.name) / "exports"
        self.exports.mkdir()
        self.picker_value = None
        self.local_ok = (True, "")
        self.pipeline = FakePipeline(self.upstream)
        self.app = create_app(self.credentials, self.upstream, self.sample, self.library, self.pipeline,
                              lambda directory=False: self.picker_value, self.history, lambda folder: self.local_ok)
        self.app.testing = True
        self.http = self.app.test_client()
        self.token = re.search(r'name="asr-token" content="([^"]+)"', self.http.get("/").get_data(as_text=True)).group(1)
        self.headers = {"X-ASR-Token": self.token}

    def tearDown(self):
        if self.upstream.hold: self.upstream.hold.set()
        for job in self.app.extensions["asr_jobs"].jobs.values():
            for _ in range(100):
                if job.status != "running": break
                time.sleep(0.01)
        self.temp.cleanup()

    def start(self, data=None):
        form = dict(data or {"use_sample": "true"})
        form.setdefault("target_language", "original")
        form.setdefault("output_dir", str(self.exports))
        form["submission_revision"] = str(self.app.extensions["asr_jobs"].revision)
        response = self.http.post("/api/transcribe", data=form, headers=self.headers)
        self.assertEqual(response.status_code, 202, response.get_data(as_text=True))
        return response.get_json()["job"]["id"]

    def wait(self, ident):
        for _ in range(200):
            job = self.http.get(f"/api/jobs/{ident}").get_json()["job"]
            if job["status"] != "running": return job
            time.sleep(0.01)
        self.fail("Fake task did not finish")

    def test_config_and_html_never_expose_key(self):
        for url in ("/", "/api/config", "/static/app.js"):
            response = self.http.get(url)
            self.assertEqual(response.status_code, 200)
            self.assertNotIn(TEST_KEY, response.get_data(as_text=True))
            response.close()
        config = self.http.get("/api/config").get_json()
        self.assertTrue(config["api_key_configured"])
        self.assertTrue(config["desktop_mode"])
        self.assertIn("zh-CN", [lang["code"] for lang in config["languages"]])
        self.assertIn("ja", [lang["code"] for lang in config["source_languages"]])
        self.assertEqual(config["default_translation"], "pro")
        self.assertEqual({m["code"] for m in config["translation_models"]}, {"pro", "flash", "deepinfra-pro", "deepinfra-flash"})
        self.assertEqual([e["code"] for e in config["asr_engines"]], ["siliconflow", "deepinfra", "local-1.7b", "local-0.6b"])
        self.assertEqual(config["keys"], {"siliconflow": True, "deepinfra": False})
        self.assertIsNone(config["last_media"])

    def test_requires_csrf_for_paid_requests(self):
        self.assertEqual(self.http.post("/api/transcribe", data={"use_sample": "true"}).status_code, 403)
        self.assertFalse(self.upstream.calls)

    def test_rejects_cross_origin_request(self):
        response = self.http.post("/api/transcribe", data={"use_sample": "true"}, headers={**self.headers, "Origin": "https://example.com"})
        self.assertEqual(response.status_code, 403)
        self.assertFalse(self.upstream.calls)

    def test_rejects_dns_rebinding_and_remote_access(self):
        self.assertEqual(self.http.get("/api/config", headers={"Host": "evil.example"}).status_code, 403)
        self.assertEqual(self.http.get("/api/config", environ_base={"REMOTE_ADDR": "192.0.2.1"}).status_code, 403)

    def test_sample_transcription_and_downloads(self):
        ident = self.start()
        job = self.wait(ident)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["text"], "こんにちは。")
        self.assertTrue(job["has_timestamps"])
        self.assertEqual(job["progress"], 100)
        self.assertEqual(self.upstream.calls[0], (b"sample-audio", "sample.mp3", TEST_KEY))
        self.assertEqual(self.http.get(f"/api/jobs/{ident}/download/txt").get_data(as_text=True), "こんにちは。")
        raw = self.http.get(f"/api/jobs/{ident}/download/json")
        self.assertEqual(json.loads(raw.data)["model"], MODEL)
        self.assertNotIn(TEST_KEY, raw.get_data(as_text=True))
        self.assertEqual(self.http.get(f"/api/jobs/{ident}/download/srt").status_code, 200)
        self.assertTrue(all(Path(file["path"]).is_file() for file in job["saved_files"]))

    def test_preserves_real_segment_timestamps(self):
        self.upstream.payload = {"text": "こんにちは。", "segments": [{"start": 1.25, "end": 2.5, "text": "こんにちは。"}]}
        ident = self.start()
        self.assertTrue(self.wait(ident)["has_timestamps"])
        self.assertIn("00:00:01,250 --> 00:00:02,500", self.http.get(f"/api/jobs/{ident}/download/srt").get_data(as_text=True))

    def test_uploaded_file(self):
        ident = self.start({"file": (BytesIO(b"uploaded-audio"), "test.wav")})
        self.assertEqual(self.wait(ident)["status"], "done")
        self.assertEqual(self.upstream.calls[0][0], b"uploaded-audio")
        self.assertEqual(list((Path(__file__).parent / "uploads").glob("*.wav")), [])

    def test_rejects_empty_invalid_and_missing_audio(self):
        for data in ({}, {"file": (BytesIO(b""), "empty.mp3")}, {"file": (BytesIO(b"abc"), "test.exe")}):
            data["submission_revision"] = "0"
            self.assertEqual(self.http.post("/api/transcribe", data=data, headers=self.headers).status_code, 400)
        self.assertFalse(self.upstream.calls)

    def test_no_api_key_no_paid_request(self):
        self.credentials["siliconflow"].clear()
        response = self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0"}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("硅基流动", response.get_json()["error"])
        self.assertFalse(self.upstream.calls)

    def test_each_cloud_provider_needs_its_own_key(self):
        for fields in ({"asr_engine": "deepinfra", "target_language": "original"},
                       {"asr_engine": "siliconflow", "target_language": "zh-CN", "translation_quality": "deepinfra-pro"}):
            response = self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0", **fields}, headers=self.headers)
            self.assertEqual(response.status_code, 400)
            self.assertIn("DeepInfra", response.get_json()["error"])
        self.assertFalse(self.upstream.calls)
        self.credentials["deepinfra"].set(DEEPINFRA_KEY)
        self.wait(self.start({"use_sample": "true", "asr_engine": "deepinfra", "target_language": "zh-CN", "translation_quality": "deepinfra-flash"}))
        options, keys = self.pipeline.runs[-1]
        self.assertEqual((options["asr_engine"], options["translation_provider"]), ("deepinfra", "deepinfra"))
        self.assertEqual(options["translation_model"], "deepseek-ai/DeepSeek-V4.1-Flash")
        self.assertEqual(keys, {"deepinfra": DEEPINFRA_KEY})

    def test_local_gpu_needs_no_key_and_reports_unavailable(self):
        self.credentials["siliconflow"].clear()
        job = self.wait(self.start({"use_sample": "true", "asr_engine": "local-1.7b", "target_language": "original"}))
        self.assertEqual(job["asr_engine"], "local-1.7b")
        self.assertTrue(job["local_asr"])
        self.assertEqual(self.pipeline.runs[-1][1], {})
        self.local_ok = (False, "本地模型未下载")
        response = self.http.post("/api/transcribe", data={"use_sample": "true", "asr_engine": "local-0.6b", "target_language": "original",
                                                         "submission_revision": str(self.app.extensions["asr_jobs"].revision)}, headers=self.headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("本地模型未下载", response.get_json()["error"])
        engines = {e["code"]: e for e in self.http.get("/api/config").get_json()["asr_engines"]}
        self.assertFalse(engines["local-0.6b"]["available"])
        self.assertTrue(engines["siliconflow"]["available"])

    def test_duplicate_request_does_not_call_provider_twice(self):
        self.upstream.hold = Event()
        ident = self.start()
        response = self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0"}, headers=self.headers)
        self.assertEqual(response.status_code, 409)
        self.upstream.hold.set()
        self.wait(ident)
        self.assertEqual(len(self.upstream.calls), 1)

    def test_stale_tab_cannot_charge_again_after_first_request_finishes(self):
        self.wait(self.start())
        response = self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0"}, headers=self.headers)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(len(self.upstream.calls), 1)

    def test_explicit_new_submission_can_run_again(self):
        self.wait(self.start())
        self.wait(self.start())
        self.assertEqual(len(self.upstream.calls), 2)

    def test_submission_revision_is_required(self):
        self.assertEqual(self.http.post("/api/transcribe", data={"use_sample": "true"}, headers=self.headers).status_code, 400)
        self.assertFalse(self.upstream.calls)

    def test_provider_error_is_redacted(self):
        self.upstream.failure = UserError("Error " + TEST_KEY)
        job = self.wait(self.start())
        self.assertEqual(job["status"], "failed")
        self.assertNotIn(TEST_KEY, job["error"])
        self.assertEqual(len(self.upstream.calls), 1)
        self.assertNotIn(TEST_KEY, next(self.history.glob("*.json")).read_text(encoding="utf-8"))

    def test_key_update_and_clear(self):
        response = self.http.post("/api/key", json={"api_key": TEST_KEY}, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertNotIn(TEST_KEY, response.get_data(as_text=True))
        self.assertEqual(self.http.post("/api/key/clear", headers=self.headers).status_code, 200)
        self.assertFalse(self.credentials["siliconflow"].get())
        self.assertEqual(self.http.post("/api/key", json={"api_key": "bad"}, headers=self.headers).status_code, 400)
        saved = self.http.post("/api/key", json={"provider": "deepinfra", "api_key": DEEPINFRA_KEY}, headers=self.headers)
        self.assertEqual(saved.status_code, 200)
        self.assertNotIn(DEEPINFRA_KEY, saved.get_data(as_text=True))
        self.assertEqual(self.credentials["deepinfra"].get(), DEEPINFRA_KEY)
        self.assertTrue(self.http.get("/api/config").get_json()["keys"]["deepinfra"])
        self.assertEqual(self.http.post("/api/key/clear", json={"provider": "deepinfra"}, headers=self.headers).status_code, 200)
        self.assertFalse(self.credentials["deepinfra"].get())
        self.assertEqual(self.http.post("/api/key", json={"provider": "other", "api_key": DEEPINFRA_KEY}, headers=self.headers).status_code, 400)

    def test_sample_preview_and_missing_sample(self):
        response = self.http.get("/api/sample/audio")
        self.assertEqual(response.data, b"sample-audio")
        response.close()
        self.sample.unlink()
        self.assertEqual(self.http.get("/api/sample/audio").status_code, 404)
        self.assertEqual(self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0"}, headers=self.headers).status_code, 404)

    def test_security_headers_and_unknown_downloads(self):
        response = self.http.get("/")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        self.assertIn("frame-ancestors 'none'", response.headers["Content-Security-Policy"])
        self.assertEqual(self.http.get("/api/jobs/unknown").status_code, 404)
        self.assertEqual(self.http.get("/api/jobs/unknown/download/txt").status_code, 404)

    def test_request_body_limit(self):
        with patch("app.MAX_AUDIO_BYTES", 8):
            response = self.http.post("/api/transcribe", data={"file": (BytesIO(b"123456789"), "large.mp3"), "submission_revision": "0"}, headers=self.headers)
            self.assertEqual(response.status_code, 413)
        self.assertFalse(self.upstream.calls)

    def test_local_video_uses_media_id_and_never_uploads_video(self):
        video = Path(self.temp.name) / "movie.mkv"
        video.write_bytes(b"local-video")
        self.picker_value = str(video)
        selected = self.http.post("/api/media/pick", headers=self.headers).get_json()["media"]
        self.assertEqual(selected["kind"], "video")
        ident = self.start({"media_id": selected["id"]})
        self.assertEqual(self.wait(ident)["status"], "done")
        self.assertEqual(video.read_bytes(), b"local-video")
        self.assertEqual(self.http.post("/api/media/pick").status_code, 403)
        self.assertEqual(self.http.get("/api/config").get_json()["last_media"]["id"], selected["id"])

    def test_picker_cancel_and_manual_path(self):
        self.assertIsNone(self.http.post("/api/media/pick", headers=self.headers).get_json()["media"])
        result = self.http.post("/api/media/load", json={"path": str(self.sample)}, headers=self.headers)
        self.assertEqual(result.status_code, 200)
        self.assertEqual(self.http.post("/api/media/load", json={}, headers=self.headers).status_code, 400)

    def test_target_language_and_bilingual_generation(self):
        ident = self.start({"use_sample": "true", "target_language": "zh-CN", "bilingual": "true"})
        job = self.wait(ident)
        self.assertEqual(job["target_language_name"], "简体中文")
        self.assertEqual(job["source_language_name"], "日语")
        self.assertEqual(job["translation_quality"], "pro")
        hinted = self.wait(self.start({"use_sample": "true", "target_language": "zh-CN", "context_hint": "访谈\n场景"}))
        self.assertEqual(hinted["context_hint"], "访谈 场景")
        self.assertEqual([(c["text"], c["original_text"]) for c in job["cues"]], [("你好。", "こんにちは。")])
        self.assertTrue(job["has_bilingual"])
        bilingual = self.http.get(f"/api/jobs/{ident}/download/bilingual").get_data(as_text=True)
        self.assertIn("こんにちは", bilingual)
        self.assertIn("你好", bilingual)
        self.assertEqual(self.http.get(f"/api/jobs/{ident}/download/original").status_code, 200)

    def test_rejects_invalid_language_and_clip_range_before_cloud_calls(self):
        for fields in ({"target_language": "unknown"}, {"source_language": "xx"}, {"translation_quality": "best"}, {"asr_engine": "cloud"},
                       {"context_hint": "长" * 201},
                       {"clip_start": "nan"}, {"clip_start": "11"}, {"clip_duration": "-1"}):
            response = self.http.post("/api/transcribe", data={"use_sample": "true", "submission_revision": "0", **fields}, headers=self.headers)
            self.assertEqual(response.status_code, 400)
        self.assertFalse(self.upstream.calls)

    def test_cancel_and_progress(self):
        self.upstream.hold = Event()
        ident = self.start()
        response = self.http.post(f"/api/jobs/{ident}/cancel", headers=self.headers)
        self.assertTrue(response.get_json()["job"]["cancel_requested"])
        self.upstream.hold.set()
        self.assertEqual(self.wait(ident)["status"], "cancelled")
        self.assertEqual(len(self.upstream.calls), 1)

    def edited_job(self):
        ident = self.start({"use_sample": "true", "target_language": "zh-CN", "bilingual": "true"})
        return ident, self.wait(ident)

    def saved(self, job, kind):
        return Path(next(f["path"] for f in job["saved_files"] if f["kind"] == kind)).read_text(encoding="utf-8")

    def test_editing_rewrites_the_jobs_own_files_and_can_be_restored(self):
        ident, job = self.edited_job()
        cues = [{"start": 0.5, "end": 1.2, "text": "你好，第一句", "original_text": "こんにちは"},
                {"start": 1.3, "end": 2.0, "text": "第二行\n手动换行", "original_text": ""}]
        response = self.http.post(f"/api/jobs/{ident}/segments", json={"cues": cues}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        edited = response.get_json()["job"]
        self.assertTrue(edited["edited"] and edited["can_restore"])
        self.assertEqual([c["text"] for c in edited["cues"]], ["你好，第一句", "第二行\n手动换行"])
        target = self.saved(edited, "srt")
        self.assertIn("00:00:01,300 --> 00:00:02,000\n第二行\n手动换行", target)
        self.assertNotIn("你好。", target)
        self.assertIn("こんにちは", self.saved(edited, "original"))
        self.assertNotIn("第二行", self.saved(edited, "original"))
        self.assertIn("你好，第一句\nこんにちは", self.saved(edited, "bilingual"))
        self.assertTrue(json.loads(self.saved(edited, "json"))["edited"])
        self.assertIn("你好，第一句", self.http.get(f"/api/jobs/{ident}/download/srt").get_data(as_text=True))
        self.assertIn(",,第二行\\N手动换行\n", self.saved(edited, "ass"))
        self.assertEqual(len(list(self.exports.iterdir())), 6)
        restored_app = create_app(self.credentials, self.upstream, self.sample, FakeMediaLibrary(), FakePipeline(self.upstream), history_dir=self.history)
        self.assertEqual(restored_app.test_client().get("/api/config").get_json()["latest_job"]["cues"][0]["text"], "你好，第一句")
        restored = self.http.post(f"/api/jobs/{ident}/segments/restore", headers=self.headers).get_json()["job"]
        self.assertFalse(restored["edited"] or restored["can_restore"])
        self.assertEqual([c["text"] for c in restored["cues"]], ["你好。"])
        self.assertIn("你好。", self.saved(restored, "srt"))
        self.assertEqual(self.http.post(f"/api/jobs/{ident}/segments/restore", headers=self.headers).status_code, 409)

    def test_style_only_save_rewrites_ass_and_becomes_the_default(self):
        config = self.http.get("/api/config").get_json()
        self.assertEqual((config["subtitle_style"], config["default_style"]), (DEFAULT_STYLE, DEFAULT_STYLE))
        self.assertIn("Microsoft YaHei", [font["code"] for font in config["subtitle_fonts"]])
        ident, job = self.edited_job()
        style = {**DEFAULT_STYLE, "color": "#FFFF00", "position": "top", "margin": 120}
        unchanged = [{"start": c["start"], "end": c["end"], "text": c["text"], "original_text": c["original_text"]} for c in job["cues"]]
        response = self.http.post(f"/api/jobs/{ident}/segments", json={"cues": unchanged, "style": style}, headers=self.headers)
        self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
        saved = response.get_json()
        self.assertEqual(saved["subtitle_style"], style)
        self.assertFalse(saved["job"]["edited"] or saved["job"]["can_restore"])
        for kind in ("ass", "bilingual_ass"):
            self.assertIn(",&H0000FFFF,", self.saved(saved["job"], kind))
            self.assertIn(",8,96,96,120,1", self.saved(saved["job"], kind))
        self.assertEqual(json.loads(self.saved(saved["job"], "json"))["style"], style)
        self.assertEqual(json.loads((Path(self.temp.name) / "settings.json").read_text(encoding="utf-8"))["subtitle_style"], style)
        restarted = create_app(self.credentials, self.upstream, self.sample, FakeMediaLibrary(), FakePipeline(self.upstream), history_dir=self.history)
        self.assertEqual(restarted.test_client().get("/api/config").get_json()["subtitle_style"], style)
        self.wait(self.start())
        self.assertEqual(self.pipeline.runs[-1][0]["style"], style)
        before = self.saved(saved["job"], "ass")
        bad = self.http.post(f"/api/jobs/{ident}/segments", json={"cues": unchanged, "style": {**style, "color": "yellow"}}, headers=self.headers)
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.saved(saved["job"], "ass"), before)

    def test_invalid_edits_never_touch_files(self):
        ident, job = self.edited_job()
        before = self.saved(job, "srt")
        good = {"start": 0.5, "end": 1.0, "text": "好", "original_text": ""}
        for cues in ([], [{**good, "text": "   "}], [{**good, "end": 0.5}], [good, {**good, "start": 0.8, "end": 1.5}],
                     [{**good, "start": "0"}], [{**good, "start": True}], [{**good, "end": float("inf")}], [{**good, "text": "长" * 1001}]):
            response = self.http.post(f"/api/jobs/{ident}/segments", data=json.dumps({"cues": cues}),
                                      headers={**self.headers, "Content-Type": "application/json"})
            self.assertEqual(response.status_code, 400, cues)
        self.assertEqual(self.http.post(f"/api/jobs/{ident}/segments", json={"cues": [good]}).status_code, 403)
        self.assertEqual(self.http.post("/api/jobs/missing/segments", json={"cues": [good]}, headers=self.headers).status_code, 404)
        self.assertEqual(self.saved(job, "srt"), before)

    def test_editing_is_refused_while_the_job_runs(self):
        self.upstream.hold = Event()
        ident = self.start()
        response = self.http.post(f"/api/jobs/{ident}/segments", json={"cues": [{"start": 0, "end": 1, "text": "x"}]}, headers=self.headers)
        self.assertEqual(response.status_code, 409)
        self.upstream.hold.set()
        self.wait(ident)

    def test_job_media_streams_the_source_with_ranges(self):
        ident, job = self.edited_job()
        response = self.http.get(f"/api/jobs/{ident}/media")
        self.assertEqual(response.data, b"sample-audio")
        response.close()
        partial = self.http.get(f"/api/jobs/{ident}/media", headers={"Range": "bytes=0-5"})
        self.assertEqual((partial.status_code, partial.data), (206, b"sample"))
        partial.close()
        self.assertEqual(self.http.get("/api/jobs/missing/media").status_code, 404)

    def test_recent_result_survives_restart_without_repeating_request(self):
        self.wait(self.start())
        count = len(self.upstream.calls)
        restored = create_app(self.credentials, self.upstream, self.sample, FakeMediaLibrary(), FakePipeline(self.upstream), history_dir=self.history)
        latest = restored.test_client().get("/api/config").get_json()["latest_job"]
        self.assertEqual(latest["status"], "done")
        self.assertEqual(len(self.upstream.calls), count)


class FormatTests(unittest.TestCase):
    def test_invalid_timestamps_cannot_create_srt(self):
        for segment in ({"start": -1, "end": 2, "text": "x"}, {"start": 0, "end": float("inf"), "text": "x"}, {"start": 2, "end": 1, "text": "x"}, {"start": True, "end": 2, "text": "x"}):
            self.assertEqual(timestamped_segments({"segments": [segment]}), [])
        self.assertEqual(timestamped_segments({"text": "hello"}), [])

    def test_srt_carries_milliseconds_correctly(self):
        self.assertIn("00:01:00,000", make_srt([{"start": 59.9996, "end": 61, "text": "x"}]))

    def test_secret_redaction(self):
        self.assertNotIn(TEST_KEY, safe_message("Authorization " + TEST_KEY, TEST_KEY))

    def test_upstream_request_is_pinned_and_does_not_retry(self):
        class Reply:
            headers = {"x-siliconcloud-trace-id": "trace"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return json.dumps({"text": "ok"}).encode()
        with patch("providers.urllib.request.build_opener") as build:
            build.return_value.open.return_value = Reply()
            payload, trace = SiliconFlowClient().transcribe(b"audio", "test.mp3", TEST_KEY)
            outbound = build.return_value.open.call_args.args[0]
            self.assertEqual(outbound.full_url, "https://api.siliconflow.cn/v1/audio/transcriptions")
            self.assertIn(MODEL.encode(), outbound.data)
            self.assertEqual(outbound.get_header("Authorization"), "Bearer " + TEST_KEY)
            self.assertEqual(payload["text"], "ok")
            self.assertEqual(trace, "trace")
            self.assertEqual(build.return_value.open.call_count, 1)

    def test_deepinfra_requests_are_pinned_and_force_language(self):
        class Reply:
            headers = {"x-request-id": "di-trace"}
            def __init__(self, payload): self.payload = payload
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return json.dumps(self.payload).encode()
        with patch("providers.urllib.request.build_opener") as build:
            build.return_value.open.return_value = Reply({"text": "", "segments": [{"text": "こん"}, {"text": "にちは"}]})
            payload, trace = DeepInfraClient().transcribe(b"audio", "speech.wav", DEEPINFRA_KEY, language="ja")
            outbound = build.return_value.open.call_args.args[0]
            self.assertEqual(outbound.full_url, "https://api.deepinfra.com/v1/inference/Qwen/Qwen3-ASR-1.7B")
            self.assertIn(b'name="language"\r\n\r\nja', outbound.data)
            self.assertIn(b'name="audio"', outbound.data)
            self.assertEqual(outbound.get_header("Authorization"), "Bearer " + DEEPINFRA_KEY)
            self.assertEqual((payload["text"], trace), ("こんにちは", "di-trace"))
            content = json.dumps({"translations": [{"id": 1, "text": "你好"}]})
            build.return_value.open.return_value = Reply({"choices": [{"finish_reason": "stop", "message": {"content": content}}]})
            mapping, _, _ = DeepInfraClient().translate([{"id": 1, "text": "こんにちは"}], "zh-CN", DEEPINFRA_KEY, model="deepseek-ai/DeepSeek-V4-Pro")
            outbound = build.return_value.open.call_args.args[0]
            sent = json.loads(outbound.data)
            self.assertEqual(outbound.full_url, DEEPINFRA_CHAT)
            self.assertEqual((sent["reasoning_effort"], sent["model"]), ("none", "deepseek-ai/DeepSeek-V4-Pro"))
            self.assertNotIn("enable_thinking", sent)
            self.assertEqual(mapping, {1: "你好"})


if __name__ == "__main__":
    unittest.main()
