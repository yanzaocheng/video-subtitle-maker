from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
import json
import subprocess
import unittest
import wave

import imageio_ffmpeg
import numpy as np

from common import Cancelled, UserError, check_cancel, is_nonverbal, language_matches
from media import Media, clip_range, extract_audio, probe_media, probabilities_to_segments
from providers import SiliconFlowClient, parse_translations
from subtitles import (DEFAULT_STYLE, SubtitlePipeline, export_files, make_ass, make_srt, normalize_style, rewrite_files,
                       wrap_subtitle)
from test_app import TEST_KEY


def write_wav(path, seconds=2):
    count = round(seconds * 16000)
    samples = (np.sin(np.arange(count) * 0.137) * 5000).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(samples.tobytes())


class FixedDetector:
    def __init__(self, intervals=None):
        self.intervals = intervals if intervals is not None else [{"start": 0.25, "end": 0.75}, {"start": 1.25, "end": 1.9}]
    def detect(self, path, progress, cancel):
        check_cancel(cancel)
        progress(1, "语音检测完成")
        return self.intervals


class CountingClient:
    def __init__(self):
        self.lock = Lock()
        self.asr_calls = 0
        self.translation_calls = 0
        self.received_audio = []
        self.failure = None
        self.translation_failure = None
        self.hold = None
        self.responder = lambda audio: "こんにちは。"
        self.translation_requests = []
        self.asr_options = []
    def transcribe(self, audio, filename, key, **options):
        with self.lock:
            self.asr_calls += 1
            self.received_audio.append(audio)
            self.asr_options.append(options)
        if self.hold: self.hold.wait(5)
        if self.failure: raise self.failure
        return {"text": self.responder(audio)}, "trace"
    def translate(self, cues, language, key, **options):
        with self.lock:
            self.translation_calls += 1
            self.translation_requests.append({"cues": [dict(c) for c in cues], "language": language, **options})
        if self.translation_failure: raise self.translation_failure
        return {c["id"]: "你好。" for c in cues}, {"prompt_tokens": 10, "completion_tokens": 5}, "translation-trace"


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source.wav"
        write_wav(self.source)
        stat = self.source.stat()
        self.media = Media("test", self.source, 2, stat.st_size, stat.st_mtime_ns, "audio")
        self.client = CountingClient()
        self.updates = {}
        self.options = {"target_language": "zh-CN", "bilingual": True, "clip_start": 0, "clip_duration": 2, "output_dir": str(self.root)}
        def extractor(media, output, start, duration, progress, cancel):
            check_cancel(cancel)
            write_wav(output, duration)
            progress(1, "音频提取完成")
            return duration
        self.pipeline = SubtitlePipeline(self.client, self.root / "cache", self.root / "work", FixedDetector(), extractor)
    def tearDown(self): self.temp.cleanup()
    def update(self, **kwargs): self.updates.update(kwargs)
    def run_pipeline(self, cancel=None): return self.pipeline.run(self.media, self.options, TEST_KEY, self.update, cancel or Event())

    def test_real_audio_segment_timestamps_are_preserved_through_translation(self):
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual(self.client.asr_calls, 2)
        self.assertEqual(self.client.translation_calls, 1)
        self.assertEqual(self.updates["progress"], 100)
        self.assertEqual(self.updates["segments"][0]["start"], 0.25)
        self.assertEqual(self.updates["segments"][1]["end"], 1.9)
        self.assertEqual(self.updates["segments"][0]["text"], "你好。")
        for audio in self.client.received_audio:
            self.assertTrue(audio.startswith(b"RIFF"))
            self.assertIn(b"WAVE", audio[:16])
        kinds = {f["kind"] for f in self.updates["saved_files"]}
        self.assertEqual(kinds, {"srt", "original", "bilingual", "ass", "bilingual_ass", "json"})
        srt = next(Path(f["path"]) for f in self.updates["saved_files"] if f["kind"] == "srt").read_text(encoding="utf-8")
        self.assertIn("00:00:00,250 --> 00:00:00,750", srt)
        self.assertNotIn("こんにちは", srt)
        self.assertEqual(list((self.root / "work").iterdir()), [])

    def test_cache_avoids_repeated_asr_and_translation_charges(self):
        self.run_pipeline()
        counts = (self.client.asr_calls, self.client.translation_calls)
        self.run_pipeline()
        self.assertEqual((self.client.asr_calls, self.client.translation_calls), counts)
        self.assertEqual(self.updates["stats"]["asr_cached"], 2)
        self.assertEqual(self.updates["stats"]["translation_cached"], 1)

    def test_original_language_never_calls_translation(self):
        self.options["target_language"] = "original"
        self.run_pipeline()
        self.assertEqual(self.client.translation_calls, 0)
        self.assertIn("こんにちは", self.updates["segments"][0]["text"])

    def test_video_clip_offset_is_added_to_srt_timestamps(self):
        self.options["clip_start"] = 300
        self.run_pipeline()
        self.assertEqual(self.updates["segments"][0]["start"], 300.25)
        self.assertIn("00:05:00,250", make_srt(self.updates["segments"]))

    def test_no_speech_makes_no_paid_requests(self):
        self.pipeline.detector = FixedDetector([])
        self.assertEqual(self.run_pipeline(), "empty")
        self.assertEqual(self.client.asr_calls, 0)
        self.assertEqual(self.client.translation_calls, 0)

    def test_pre_cancelled_job_makes_no_paid_requests(self):
        cancel = Event()
        cancel.set()
        with self.assertRaises(Cancelled): self.run_pipeline(cancel)
        self.assertEqual(self.client.asr_calls, 0)

    def test_fail_fast_stops_queued_requests(self):
        self.pipeline.detector = FixedDetector([{"start": 0.01 * i, "end": 0.01 * i + 0.005} for i in range(30)])
        self.client.failure = UserError("provider failed")
        with self.assertRaises(UserError): self.run_pipeline()
        self.assertLessEqual(self.client.asr_calls, 4)
        self.assertEqual(self.client.translation_calls, 0)

    def test_translation_failure_keeps_original_segments_and_asr_cache(self):
        self.client.translation_failure = UserError("translation failed")
        with self.assertRaises(UserError): self.run_pipeline()
        self.assertEqual(len(self.updates["original_segments"]), 2)
        previous = self.client.asr_calls
        self.client.translation_failure = None
        self.run_pipeline()
        self.assertEqual(self.client.asr_calls, previous)

    def test_wrong_language_output_is_filtered_before_translation(self):
        self.options["source_language"] = "ja"
        self.client.responder = lambda audio: "nên để" if len(audio) < 18000 else "こんにちは。"
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual(self.updates["stats"]["filtered_segments"], 1)
        self.assertEqual([c["text"] for c in self.client.translation_requests[0]["cues"]], ["こんにちは。"])
        saved = "".join(Path(f["path"]).read_text(encoding="utf-8") for f in self.updates["saved_files"] if f["kind"] != "json")
        self.assertNotIn("nên", saved)

    def test_nonverbal_sounds_are_skipped_only_when_enabled(self):
        self.options.update(source_language="ja", skip_nonverbal=True)
        self.client.responder = lambda audio: "あっ、あぁ…んっ" if len(audio) < 18000 else "気持ちいい。"
        self.run_pipeline()
        self.assertEqual(self.updates["stats"]["nonverbal_skipped"], 1)
        self.assertEqual(len(self.updates["segments"]), 1)
        self.options["skip_nonverbal"] = False
        self.run_pipeline()
        self.assertEqual(len(self.updates["segments"]), 2)

    def test_translation_gets_neighbour_context_model_and_language(self):
        self.options.update(source_language="ja", translation_model="test/model", clip_duration=2, context_hint="访谈")
        self.pipeline.detector = FixedDetector([{"start": i * 0.07, "end": i * 0.07 + 0.05} for i in range(25)])
        self.client.responder = lambda audio: f"第{len(audio)}。"
        self.run_pipeline()
        self.assertEqual(self.client.translation_calls, 2)
        first, second = sorted(self.client.translation_requests, key=lambda r: r["cues"][0]["start"])
        self.assertEqual(second["context_before"], [c["text"] for c in first["cues"][-6:]])
        self.assertEqual(first["context_after"], [c["text"] for c in second["cues"][:3]])
        self.assertEqual({r["model"] for r in self.client.translation_requests}, {"test/model"})
        self.assertEqual({r["source_language"] for r in self.client.translation_requests}, {"ja"})
        self.assertEqual({r["context_hint"] for r in self.client.translation_requests}, {"访谈"})
        self.assertEqual([c["id"] for c in self.updates["segments"]], list(range(1, 26)))

    def test_translation_cache_is_separate_per_model_and_hint(self):
        self.options["translation_model"] = "model-a"
        self.run_pipeline()
        self.options["translation_model"] = "model-b"
        self.run_pipeline()
        self.options["context_hint"] = "新的背景"
        self.run_pipeline()
        self.assertEqual(self.client.translation_calls, 3)
        self.assertEqual(self.client.asr_calls, 2)

    def test_translation_prompt_includes_hint_and_homophone_rule(self):
        class Reply:
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{"translations":[{"id":1,"text":"你好"}]}'}}]}).encode()
        from unittest.mock import patch
        with patch("providers.urllib.request.build_opener") as build:
            build.return_value.open.return_value = Reply()
            mapping, _, _ = SiliconFlowClient().translate([{"id": 1, "text": "こんにちは"}], "zh-CN", TEST_KEY, source_language="ja",
                                                      model="m", context_before=["前"], context_after=["后"], context_hint="访谈")
            sent = json.loads(build.return_value.open.call_args.args[0].data)
        self.assertEqual(mapping, {1: "你好"})
        self.assertEqual(sent["model"], "m")
        system = sent["messages"][0]["content"]
        self.assertIn("日语字幕翻译为简体中文", system)
        self.assertIn("同音", system)
        self.assertIn("访谈", system)
        user = json.loads(sent["messages"][1]["content"])
        self.assertEqual((user["context_before"], user["context_after"]), (["前"], ["后"]))

    def test_deepinfra_uses_its_own_cache_and_forces_language(self):
        self.options.update(source_language="ja")
        self.run_pipeline()
        self.options["asr_engine"] = "deepinfra"
        self.run_pipeline()
        self.assertEqual(self.client.asr_calls, 4)
        self.assertEqual({o.get("language") for o in self.client.asr_options}, {"ja"})
        self.run_pipeline()
        self.assertEqual(self.client.asr_calls, 4)

    def test_missing_key_stops_before_any_work(self):
        extracted = []
        self.pipeline.extractor = lambda *args: extracted.append(args)
        self.options["asr_engine"] = "deepinfra"
        with self.assertRaises(UserError):
            self.pipeline.run(self.media, self.options, {"siliconflow": TEST_KEY}, self.update, Event())
        self.assertEqual((extracted, self.client.asr_calls), ([], 0))

    def test_local_gpu_engine_batches_and_always_closes(self):
        class FakeEngine:
            def __init__(self):
                self.batch_size, self.batches, self.started, self.closed, self.failure = 1, [], False, False, None
            def start(self, cancel, progress):
                self.started = True
                progress("本地模型已加载")
            def transcribe_batch(self, audios, language, cancel):
                if self.failure: raise self.failure
                self.batches.append((len(audios), language))
                return ["本地の声。"] * len(audios)
            def close(self):
                self.closed = True
        engine = FakeEngine()
        engine.batch_size = 2
        self.pipeline.local_factory = lambda folder: engine
        self.pipeline.detector = FixedDetector([{"start": i * 0.3, "end": i * 0.3 + 0.2} for i in range(5)])
        self.options.update(asr_engine="local-1.7b", source_language="ja", target_language="original")
        self.assertEqual(self.pipeline.run(self.media, self.options, {}, self.update, Event()), "done")
        self.assertTrue(engine.started and engine.closed)
        self.assertEqual(engine.batches, [(2, "ja"), (2, "ja"), (1, "ja")])
        self.assertEqual((self.client.asr_calls, self.updates["stats"]["uploaded_seconds"]), (0, 0))
        self.assertEqual(len(self.updates["segments"]), 5)
        failing = FakeEngine()
        failing.failure = UserError("显存不足")
        self.pipeline.local_factory = lambda folder: failing
        self.pipeline.detector = FixedDetector([{"start": 0.1, "end": 0.4}])
        with self.assertRaises(UserError):
            self.pipeline.run(self.media, self.options, {}, self.update, Event())
        self.assertTrue(failing.closed)

    def test_export_writes_styled_ass_files(self):
        self.options["style"] = {**DEFAULT_STYLE, "color": "#FFCC00"}
        self.run_pipeline()
        files = {f["kind"]: Path(f["path"]) for f in self.updates["saved_files"]}
        self.assertTrue(files["ass"].name.endswith(".zh-CN.ass"))
        self.assertTrue(files["bilingual_ass"].name.endswith(".zh-CN.bilingual.ass"))
        self.assertTrue(files["ass"].read_bytes().startswith(b"\xef\xbb\xbf[Script Info]"))
        target = files["ass"].read_text(encoding="utf-8-sig")
        self.assertIn(",&H0000CCFF,", target)
        self.assertIn("Dialogue: 0,0:00:00.25,0:00:00.75,Default,,0,0,0,,你好。", target)
        self.assertNotIn("こんにちは", target)
        self.assertIn("你好。\\N{\\fs", files["bilingual_ass"].read_text(encoding="utf-8-sig"))
        self.assertEqual(json.loads(files["json"].read_text(encoding="utf-8"))["style"]["color"], "#FFCC00")

    def test_auto_export_never_overwrites_existing_subtitles(self):
        self.run_pipeline()
        original_paths = [Path(f["path"]) for f in self.updates["saved_files"]]
        original_contents = [p.read_bytes() for p in original_paths]
        self.run_pipeline()
        self.assertTrue(set(original_paths).isdisjoint(Path(f["path"]) for f in self.updates["saved_files"]))
        self.assertEqual([p.read_bytes() for p in original_paths], original_contents)


class TimingAndTranslationTests(unittest.TestCase):
    def test_vad_does_not_distribute_text_over_silence(self):
        probabilities = [0.0] * 100 + [0.9] * 30 + [0.0] * 100
        result = probabilities_to_segments(probabilities, len(probabilities) * 0.032)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["start"], 3.12, places=4)
        self.assertLess(result[0]["end"], 4.5)

    def test_long_speech_is_split_without_overlaps(self):
        result = probabilities_to_segments([0.9] * 1000, 32)
        self.assertGreater(len(result), 1)
        for left, right in zip(result, result[1:]): self.assertLessEqual(left["end"], right["start"])
        self.assertLessEqual(result[-1]["end"], 32)

    def test_short_pauses_are_merged_into_sentences(self):
        speech, short_pause, long_pause = [0.9] * 31, [0.0] * 16, [0.0] * 38
        merged = probabilities_to_segments(speech + short_pause + speech + long_pause, 4.0)
        self.assertEqual(len(merged), 1)
        separate = probabilities_to_segments(speech + long_pause + speech + long_pause, 5.0)
        self.assertEqual(len(separate), 2)
        too_long = probabilities_to_segments(([0.9] * 220 + short_pause) * 2, 15.0)
        self.assertEqual(len(too_long), 2)
        self.assertLessEqual(max(s["end"] - s["start"] for s in too_long), 10.5)

    def test_language_and_nonverbal_rules(self):
        self.assertTrue(language_matches("こんにちは", "ja"))
        self.assertTrue(language_matches("東京", "ja"))
        self.assertFalse(language_matches("dans des.", "ja"))
        self.assertFalse(language_matches("nên để", "ja"))
        self.assertTrue(language_matches("hello", "en"))
        self.assertFalse(language_matches("こんにちは", "en"))
        self.assertTrue(language_matches("anything", "auto"))
        for text in ("あっ、あぁ…", "んっ、んー", "はぁ、はぁ", "アアッ！", "…。"):
            self.assertTrue(is_nonverbal(text), text)
        for text in ("うん。", "ええ", "はい", "いい…", "イクっ", "やだ", "あっ、気持ちいい"):
            self.assertFalse(is_nonverbal(text), text)

    def test_wrapping_keeps_manual_line_breaks(self):
        self.assertEqual(wrap_subtitle("第一行\n  第二行  \n\n第三行"), "第一行\n第二行\n第三行")

    def test_wrapping_never_starts_a_line_with_closing_punctuation(self):
        for text in ("一" * 24 + "。", "一" * 23 + "，完全没有。"):
            for line in wrap_subtitle(text).split("\n"):
                self.assertNotIn(line[0], "。，！？、")

    def test_ass_follows_style_and_video_aspect(self):
        style = {"font": "SimHei", "size": 60, "color": "#ffcc00", "outline_color": "#112233", "border": "outline",
                 "position": "top", "margin": 80}
        text = make_ass([{"start": 1.254, "end": 3723.5, "text": "你好"}], style, (1440, 1080))
        self.assertIn("PlayResX: 1440\nPlayResY: 1080", text)
        self.assertIn("Style: Default,SimHei,60,&H0000CCFF,&H000000FF,&H00332211,&H00000000,0,0,0,0,100,100,0,0,1,3.3,0,8,72,72,80,1", text)
        self.assertIn("Dialogue: 0,0:00:01.25,1:02:03.50,Default,,0,0,0,,你好", text)
        # YaHei is sized by its 1.32 em cell in ASS renderers; the box style pads with a translucent outline colour.
        boxed = make_ass([{"start": 0, "end": 1, "text": "x"}], {**DEFAULT_STYLE, "border": "box"})
        self.assertIn("Style: Default,Microsoft YaHei,71.3,&H00FFFFFF,&H000000FF,&H66000000,&H00000000,0,0,0,0,100,100,0,0,3,6.5,0,2,96,96,50,1", boxed)
        self.assertIn(",100,100,0,0,1,0,0,2,96,96,50,1", make_ass([], {**DEFAULT_STYLE, "border": "none", "size": 60}))

    def test_ass_escapes_tags_and_wraps_for_the_font_size(self):
        long_text = "一" * 40
        normal = make_ass([{"start": 0, "end": 1, "text": long_text}])
        large = make_ass([{"start": 0, "end": 1, "text": long_text}], {**DEFAULT_STYLE, "size": 108})
        self.assertEqual(normal.split(",,")[-1].count("\\N"), 1)
        self.assertEqual(large.split(",,")[-1].count("\\N"), 2)
        escaped = make_ass([{"start": 0, "end": 1, "text": "{\\an8}测试\n第二行"}])
        self.assertIn(",,｛＼an8｝测试\\N第二行\n", escaped)
        bilingual = make_ass([{"start": 0, "end": 1, "text": "你好", "original_text": "こんにちは"},
                              {"start": 1, "end": 2, "text": "只有译文", "original_text": " "}], bilingual=True)
        self.assertIn(",,你好\\N{\\fs53.5\\bord2.2}こんにちは\n", bilingual)
        self.assertIn(",,只有译文\n", bilingual)

    def test_style_validation(self):
        self.assertEqual(normalize_style({**DEFAULT_STYLE, "color": "#abcdef", "size": 60.04}),
                         {**DEFAULT_STYLE, "color": "#ABCDEF", "size": 60})
        for change in ({"color": "red"}, {"size": 500}, {"size": True}, {"margin": float("nan")}, {"font": "Comic Sans"},
                       {"border": "glow"}, {"position": "left"}, {"outline_color": None}):
            with self.assertRaises(UserError):
                normalize_style({**DEFAULT_STYLE, **change})
            self.assertEqual(normalize_style({**DEFAULT_STYLE, **change}, strict=False), DEFAULT_STYLE)
        with self.assertRaises(UserError):
            normalize_style("bold")
        self.assertEqual(normalize_style(None, strict=False), DEFAULT_STYLE)

    def test_rewrite_adds_ass_files_to_jobs_saved_before_they_existed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            media = Media("x", root / "movie.mkv", 10, 1, 0, "video")
            cues = [{"start": 0, "end": 1, "text": "你好", "original_text": "こんにちは"}]
            original = [{"start": 0, "end": 1, "text": "こんにちは"}]
            saved = export_files(media, "zh-CN", original, cues, True, root, {"old": True})
            old = [item for item in saved if "ass" not in item["kind"]]
            for item in saved:
                if "ass" in item["kind"]:
                    Path(item["path"]).unlink()
            (root / "movie.zh-CN.ass").write_text("user file", encoding="utf-8")
            updated = rewrite_files(old, "zh-CN", original, cues, True, {"edited": True}, {**DEFAULT_STYLE, "position": "top"})
            self.assertEqual([item["kind"] for item in updated], ["srt", "original", "bilingual", "ass", "bilingual_ass", "json"])
            paths = {item["kind"]: Path(item["path"]) for item in updated}
            self.assertEqual((paths["ass"].name, paths["bilingual_ass"].name), ("movie.zh-CN.1.ass", "movie.zh-CN.bilingual.ass"))
            self.assertEqual((root / "movie.zh-CN.ass").read_text(encoding="utf-8"), "user file")
            self.assertIn(",8,96,96,50,1", paths["ass"].read_text(encoding="utf-8-sig"))
            self.assertEqual(json.loads(paths["json"].read_text(encoding="utf-8")), {"old": True, "edited": True})

    def test_silence_produces_no_segments(self):
        self.assertEqual(probabilities_to_segments([0.0] * 100, 3.2), [])

    def test_overlapping_and_nonfinite_srt_are_rejected(self):
        with self.assertRaises(UserError): make_srt([{"start": 1, "end": 3, "text": "a"}, {"start": 2, "end": 4, "text": "b"}])
        with self.assertRaises(UserError): make_srt([{"start": 0, "end": float("inf"), "text": "x"}])

    def test_translation_ids_can_be_reordered_but_not_lost_or_duplicated(self):
        self.assertEqual(parse_translations('{"translations":[{"id":2,"text":"乙"},{"id":1,"text":"甲"}]}', [1, 2]), {2: "乙", 1: "甲"})
        for content in ('{"translations":[]}', '{"translations":[{"id":1,"text":"甲"},{"id":1,"text":"乙"}]}', '{"translations":[{"id":true,"text":"甲"},{"id":2,"text":"乙"}]}', 'not json'):
            with self.assertRaises(UserError): parse_translations(content, [1, 2])

    def test_file_paths_and_clip_ranges_are_validated(self):
        with self.assertRaises(UserError): probe_media("https://example.com/movie.mp4")
        with self.assertRaises(UserError): probe_media(r"\\server\share\movie.mp4")
        media = Media("x", Path("test.mp4"), 100, 1, 0, "video")
        self.assertEqual(clip_range(media, 90, 30), (90, 10))
        for start, duration in ((-1, 2), (100, 1), (0, float("nan")), (float("inf"), 1)):
            with self.assertRaises(UserError): clip_range(media, start, duration)


FAKE_WORKER = r'''
import json, sys
print("warning text that is not json", flush=True)
print(json.dumps({"ready": True, "gpu": "Fake GPU", "free_after_mb": 1234}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    audios = request["audios"]
    if len(audios) > 2:
        print(json.dumps({"id": request["id"], "oom": True}), flush=True)
    elif request.get("language") == "xx":
        print(json.dumps({"id": request["id"], "error": "boom"}), flush=True)
    elif request.get("language") == "hang":
        import time; time.sleep(30)
    else:
        print(json.dumps({"id": request["id"], "texts": ["t%d" % len(a) for a in audios]}), flush=True)
'''


class LocalWorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        import sys
        from local_asr import LocalAsrEngine
        self.temp = TemporaryDirectory()
        root = Path(self.temp.name)
        (root / "models" / "Fake").mkdir(parents=True)
        (root / "models" / "Fake" / "verified.json").write_text("{}")
        (root / "models" / "Fake" / "model.safetensors").write_bytes(b"")
        (root / "worker.py").write_text(FAKE_WORKER, encoding="utf-8")
        self.engine = LocalAsrEngine("Fake", batch_size=4, python=sys.executable, worker=root / "worker.py",
                                     models=root / "models", log_dir=root / "logs")
        self.stages = []
        self.engine.start(Event(), self.stages.append)

    def tearDown(self):
        self.engine.close()
        self.temp.cleanup()

    def test_out_of_memory_shrinks_batches_and_keeps_order(self):
        audios = [b"a" * n for n in range(1, 6)]
        texts = self.engine.transcribe_batch(audios, "ja", Event())
        import base64
        self.assertEqual(texts, ["t%d" % len(base64.b64encode(a)) for a in audios])
        self.assertEqual(self.engine.batch_size, 2)
        self.assertIn("Fake GPU", self.stages[-1])

    def test_worker_error_is_reported(self):
        with self.assertRaises(UserError):
            self.engine.transcribe_batch([b"x"], "xx", Event())

    def test_cancel_stops_waiting_and_kills_worker(self):
        cancel = Event()
        from threading import Timer
        Timer(0.5, cancel.set).start()
        with self.assertRaises(Cancelled):
            self.engine.transcribe_batch([b"x"], "hang", cancel)
        self.assertIsNone(self.engine.process)

    def test_missing_model_is_reported_before_starting(self):
        from local_asr import LocalAsrEngine
        engine = LocalAsrEngine("Missing", python=self.engine.python, worker=self.engine.worker, models=self.engine.models)
        with self.assertRaises(UserError):
            engine.start(Event(), lambda label: None)
        self.assertIsNone(engine.process)


class VideoExtractionTests(unittest.TestCase):
    def test_delayed_audio_keeps_video_timeline_before_and_after_crop(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "delayed.mkv"
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=5", "-itsoffset", "2", "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=2", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video)]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            media = probe_media(video)
            for start, expected_silence in ((0, 2), (1, 1)):
                output = root / f"offset-{start}.wav"
                extract_audio(media, output, start, 4 - start, lambda *args: None, Event())
                with wave.open(str(output), "rb") as audio:
                    pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
                self.assertGreaterEqual(len(pcm), round((expected_silence + 1.5) * 16000))
                self.assertEqual(int(np.max(np.abs(pcm[:round((expected_silence - 0.1) * 16000)]))), 0)
                self.assertGreater(float(np.mean(np.abs(pcm[round((expected_silence + 0.1) * 16000):]))), 100)

    def test_audio_timestamp_gaps_are_not_collapsed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, output = root / "gap.mkv", root / "gap.wav"
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=6", "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=6", "-af", r"aselect=not(between(t\,2\,4))", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video)]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            extract_audio(probe_media(video), output, 0, 6, lambda *args: None, Event())
            with wave.open(str(output), "rb") as audio:
                pcm = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
            self.assertGreaterEqual(len(pcm), 5.9 * 16000)
            self.assertEqual(int(np.max(np.abs(pcm[round(2.5 * 16000):round(3.5 * 16000)]))), 0)
            self.assertGreater(float(np.mean(np.abs(pcm[5 * 16000:]))), 100)

    def test_local_video_extraction_preserves_source_and_duration(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            video, output = root / "video.mkv", root / "speech.wav"
            command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-n", "-f", "lavfi", "-i", "color=c=black:s=64x64:r=5:d=3", "-f", "lavfi", "-i", "sine=frequency=500:sample_rate=16000:duration=3", "-c:v", "mpeg4", "-c:a", "pcm_s16le", str(video)]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)
            before = video.read_bytes()
            media = probe_media(video)
            self.assertEqual(media.kind, "video")
            stages = []
            actual = extract_audio(media, output, 0.5, 2, lambda ratio, stage: stages.append((ratio, stage)), Event())
            self.assertAlmostEqual(actual, 2, delta=0.02)
            self.assertEqual(video.read_bytes(), before)
            self.assertEqual(stages[-1][0], 1)
            with wave.open(str(output), "rb") as audio:
                self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getsampwidth()), (1, 16000, 2))


if __name__ == "__main__": unittest.main()
