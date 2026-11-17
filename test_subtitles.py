from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock
from unittest.mock import patch
import json
import subprocess
import time
import unittest
import urllib.error
import wave

import imageio_ffmpeg
import numpy as np

from common import Cancelled, UserError, check_cancel, is_filler, is_nonverbal, language_matches
from media import Media, SpeechDetector, clip_range, extract_audio, probe_media, probabilities_to_segments
from providers import (ContentError, CustomClient, SiliconFlowClient, custom_endpoint, parse_translation_payload,
                       parse_translations)
from subtitles import (DEFAULT_STYLE, REFUSED_LINES_LIMIT, SubtitlePipeline, export_files, make_ass, make_srt, normalize_style,
                       rewrite_files, wrap_subtitle)
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
        # Decides per request whether the provider turns the text down: refuses(cues, context_before, context_after).
        self.refuses = lambda cues, before, after: False
        self.keys = []
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
            self.keys.append(key)
        if self.translation_failure: raise self.translation_failure
        if self.refuses(cues, options.get("context_before", ()), options.get("context_after", ())):
            raise ContentError("服务商 HTTP 451：prohibited content")
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

    def test_sentences_found_on_the_second_listen_must_have_content(self):
        # The short clip is the first range, the long one the second.
        self.options.update(source_language="ja", target_language="original")
        self.client.responder = lambda audio: "うん。" if len(audio) < 18000 else "先に入ってて。"
        self.pipeline.detector = FixedDetector([{"start": 0.25, "end": 0.75, "recovered": True}, {"start": 1.25, "end": 1.9}])
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual([c["text"] for c in self.updates["segments"]], ["先に入ってて。"])
        self.assertEqual((self.updates["stats"]["recovered_skipped"], self.updates["stats"]["recovered_segments"]), (1, 0))
        saved = next(Path(f["path"]) for f in self.updates["saved_files"] if f["kind"] == "json")
        removed = json.loads(saved.read_text(encoding="utf-8"))["removed_segments"]
        self.assertEqual([(r["text"], "复查" in r["reason"]) for r in removed], [("うん。", True)])
        # The same answer is kept where the detector was sure of it, and a real sentence is kept either way.
        self.pipeline.detector = FixedDetector([{"start": 0.25, "end": 0.75}, {"start": 1.25, "end": 1.9, "recovered": True}])
        self.run_pipeline()
        self.assertEqual([c["text"] for c in self.updates["segments"]], ["うん。", "先に入ってて。"])
        self.assertEqual((self.updates["stats"]["recovered_skipped"], self.updates["stats"]["recovered_segments"]), (0, 1))
        self.assertTrue(all("recovered" not in cue for cue in self.updates["segments"]))
        # Sounds are dropped there even when the option to skip them is off.
        self.options["skip_nonverbal"] = False
        self.client.responder = lambda audio: "あっ、あぁ…" if len(audio) < 18000 else "先に入ってて。"
        self.pipeline.detector = FixedDetector([{"start": 0.25, "end": 0.75, "recovered": True}, {"start": 1.25, "end": 1.9}])
        self.run_pipeline()
        self.assertEqual((len(self.updates["segments"]), self.updates["stats"]["recovered_skipped"]), (1, 1))

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

    def many_lines(self, count):
        step = 1.9 / count  # the test audio is 2 seconds long
        self.pipeline.detector = FixedDetector([{"start": i * step, "end": i * step + step * 0.7} for i in range(count)])
        self.options.update(source_language="ja")

    def test_refused_lines_keep_the_original_and_the_rest_is_translated(self):
        self.many_lines(45)
        refused = {3, 4, 28}
        self.client.refuses = lambda cues, before, after: any(c["id"] in refused for c in cues)
        self.assertEqual(self.run_pipeline(), "done")
        segments = self.updates["segments"]
        self.assertEqual([c["id"] for c in segments], list(range(1, 46)))
        self.assertEqual({c["id"] for c in segments if c["text"] == c["original_text"]}, refused)
        self.assertTrue(all(c["text"] == "你好。" for c in segments if c["id"] not in refused))
        self.assertEqual(self.updates["stats"]["untranslated"], 3)
        self.assertIn("有 3 句没有翻译成功", self.updates["warning"])
        self.assertIn("HTTP 451", self.updates["warning"])
        self.assertIn("3 句未翻译", self.updates["stage"])
        files = {f["kind"]: Path(f["path"]).read_text(encoding="utf-8-sig") for f in self.updates["saved_files"]}
        self.assertEqual(json.loads(files["json"])["untranslated_ids"], [3, 4, 28])
        # An untranslated line is not printed twice in the bilingual files.
        self.assertEqual(files["bilingual"].count("こんにちは。"), 45)
        self.assertEqual(files["bilingual_ass"].count("こんにちは。"), 45)
        # The parts that were refused go without neighbour context; nothing that worked is asked again on a rerun.
        self.assertTrue(all(not r["context_before"] and not r["context_after"] for r in self.client.translation_requests if len(r["cues"]) < 5))
        first = self.client.translation_calls
        self.assertLess(first, 30)
        self.run_pipeline()
        self.assertLess(self.client.translation_calls - first, 15)
        self.assertEqual(self.updates["stats"]["untranslated"], 3)

    def test_a_line_refused_only_for_its_context_is_translated_alone(self):
        self.many_lines(21)
        self.client.refuses = lambda cues, before, after: len(cues) == 1 and bool(before)
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual(self.updates["stats"]["untranslated"], 0)
        self.assertEqual(self.updates["segments"][-1]["text"], "你好。")
        self.assertNotIn("没有翻译成功", self.updates.get("warning", ""))

    def test_a_service_that_refuses_everything_is_given_up_on(self):
        self.many_lines(120)
        self.client.refuses = lambda cues, before, after: True
        with self.assertRaises(UserError) as raised:
            self.run_pipeline()
        self.assertIn("拒绝了大部分内容", str(raised.exception))
        # About one batch taken apart line by line on each of the three workers, not all 120 lines.
        self.assertLessEqual(self.client.translation_calls, 4 * REFUSED_LINES_LIMIT)
        self.assertEqual(len(self.updates["original_segments"]), 120)

    def test_many_refused_lines_do_not_stop_a_service_that_translates_the_rest(self):
        self.many_lines(120)
        refused = set(range(1, 121, 3))
        self.client.refuses = lambda cues, before, after: any(c["id"] in refused for c in cues)
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual(self.updates["stats"]["untranslated"], 40)
        # A rerun finds the translated lines in the cache and asks only for the refused ones again.
        first = self.client.translation_calls
        self.assertEqual(self.run_pipeline(), "done")
        self.assertEqual(self.updates["stats"]["untranslated"], 40)
        self.assertLess(self.client.translation_calls - first, first)

    def test_custom_api_gets_only_its_own_key_and_its_own_cache(self):
        made, formats = [], []
        def factory(endpoint, api_format="chat"):
            made.append(endpoint)
            formats.append(api_format)
            return self.client
        self.pipeline.custom_factory = factory
        self.options.update(translation_provider="custom", translation_model="my-model",
                            translation_endpoint="https://llm.example/v1/chat/completions")
        keys = {"siliconflow": TEST_KEY, "custom": "custom-key"}
        self.assertEqual(self.pipeline.run(self.media, self.options, keys, self.update, Event()), "done")
        self.assertEqual((made, self.client.keys), (["https://llm.example/v1/chat/completions"], ["custom-key"]))
        self.assertEqual(self.client.translation_requests[0]["model"], "my-model")
        metadata = json.loads(next(Path(f["path"]) for f in self.updates["saved_files"] if f["kind"] == "json").read_text(encoding="utf-8"))
        self.assertEqual((metadata["translation_provider"], metadata["translation_endpoint"]), ("custom", "https://llm.example/v1/chat/completions"))
        # Without a saved custom key nothing is sent as a key, whatever other keys the job has.
        self.options["translation_endpoint"] = "http://localhost:11434/v1/chat/completions"
        self.pipeline.run(self.media, self.options, {"siliconflow": TEST_KEY}, self.update, Event())
        self.pipeline.run(self.media, self.options, TEST_KEY, self.update, Event())
        self.assertEqual(self.client.keys, ["custom-key", ""])
        self.assertEqual(self.client.translation_calls, 2)
        # The format the user chose goes to the client; the other endpoint has its own cache.
        self.options.update(translation_endpoint="https://llm.example/v1/responses", translation_format="responses")
        self.pipeline.run(self.media, self.options, keys, self.update, Event())
        self.assertEqual((formats[0], formats[-1], made[-1], self.client.translation_calls), ("chat", "responses", "https://llm.example/v1/responses", 3))

    def test_local_model_loads_while_the_audio_is_prepared(self):
        events, release = [], Event()
        class SlowEngine:
            batch_size = 4
            def start(self, cancel, progress):
                events.append("load started")
                progress("正在加载模型")
                try:
                    while not release.wait(0.01):
                        check_cancel(cancel)
                except Cancelled:
                    events.append("load stopped")
                    raise
                events.append("load finished")
            def transcribe_batch(self, audios, language, cancel):
                events.append("recognise")
                return ["本地の声。"] * len(audios)
            def close(self):
                events.append("closed")
        class Detector(FixedDetector):
            def detect(self, path, progress, cancel):
                for _ in range(200):
                    if events: break
                    time.sleep(0.01)
                events.append("speech detected")
                release.set()
                return super().detect(path, progress, cancel)
        self.pipeline.local_factory = lambda folder: SlowEngine()
        self.pipeline.detector = Detector()
        self.options.update(asr_engine="local-1.7b", source_language="ja", target_language="original")
        self.assertEqual(self.pipeline.run(self.media, self.options, {}, self.update, Event()), "done")
        self.assertEqual(events[:2], ["load started", "speech detected"])
        self.assertEqual((events.index("load finished") < events.index("recognise"), events[-1]), (True, "closed"))
        # A job that ends before recognition stops the loading instead of leaving the model on the card.
        events.clear()
        release.clear()
        def failing(media, output, start, duration, progress, cancel):
            for _ in range(200):
                if events: break
                time.sleep(0.01)
            raise UserError("提取失败")
        self.pipeline.extractor = failing
        with self.assertRaises(UserError) as raised:
            self.pipeline.run(self.media, self.options, {}, self.update, Event())
        self.assertEqual((str(raised.exception), events), ("提取失败", ["load started", "load stopped", "closed"]))

    def test_translation_prompt_includes_hint_and_homophone_rule(self):
        class Reply:
            headers = {}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self, limit): return json.dumps({"choices": [{"finish_reason": "stop", "message": {"content": '{"translations":[{"id":1,"text":"你好"}]}'}}]}).encode()
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

    def test_lines_without_content_are_recognised(self):
        for text in ("うん。", "はい。", "うん、うん。", "うーん", "はい、はい。", "えっと。", "なんか。", "あ、あの。", "そうそうそう。",
                     "か。", "で。", "あっ、あぁ…", "…。", "Yeah.", "Oh, okay.", "Mmm", "嗯嗯。", "好。"):
            self.assertTrue(is_filler(text), text)
        for text in ("先に入ってて。", "なんで隠すんだ。", "ああ、そうです。", "最高。", "見て。", "はい、お願いします。", "うん、行く。",
                     "Yeah, right there.", "No.", "好的。"):
            self.assertFalse(is_filler(text), text)

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

    def test_answers_wrapped_in_thoughts_or_prose_are_still_read(self):
        answer = '{"translations":[{"id":1,"text":"甲"}]}'
        for content in (answer, f"<think>先想一想 {{}} </think>\n{answer}", f"```json\n{answer}\n```", f"好的，结果如下：\n{answer}\n希望有帮助。"):
            self.assertEqual(parse_translations(content, [1]), {1: "甲"}, content)
        for content in ("", "```", "没有 JSON", None, '{"translations":[{"id":1,"text":"  "}]}'):
            with self.assertRaises(ContentError): parse_translations(content, [1])
        reply = lambda reason: {"choices": [{"finish_reason": reason, "message": {"content": answer}}]}
        # Built-in providers must say "stop"; a custom server may name a normal end differently, but never a cut answer.
        self.assertEqual(parse_translation_payload(reply("stop"), [1]), {1: "甲"})
        self.assertEqual(parse_translation_payload(reply("eos"), [1], None), {1: "甲"})
        for reason, allowed in (("eos", (None, "stop")), ("length", None), ("content_filter", None)):
            with self.assertRaises(ContentError): parse_translation_payload(reply(reason), [1], allowed)
        with self.assertRaises(UserError): parse_translation_payload({"choices": ["x"]}, [1], None)

    def test_custom_endpoint_accepts_base_urls_and_full_addresses(self):
        for given, expected in (("https://api.deepseek.com/v1", "https://api.deepseek.com/v1/chat/completions"),
                                (" https://api.deepseek.com/v1/ ", "https://api.deepseek.com/v1/chat/completions"),
                                ("https://api.deepseek.com", "https://api.deepseek.com/v1/chat/completions"),
                                ("http://localhost:11434/v1/chat/completions", "http://localhost:11434/v1/chat/completions"),
                                ("https://x.example/openai/chat/completions?api-version=1", "https://x.example/openai/chat/completions?api-version=1")):
            self.assertEqual(custom_endpoint(given), expected)
        # Each format has its own path; a bare host gets /v1, an address copied for another format is switched over.
        for given, api_format, expected in (("https://gw.example/v1", "responses", "https://gw.example/v1/responses"),
                                            ("https://gw.example/v1/", "messages", "https://gw.example/v1/messages"),
                                            ("https://api.anthropic.com", "messages", "https://api.anthropic.com/v1/messages"),
                                            ("https://gw.example/v1/chat/completions", "responses", "https://gw.example/v1/responses"),
                                            ("https://gw.example/v1/responses", "chat", "https://gw.example/v1/chat/completions"),
                                            ("https://gw.example/api/messages", "messages", "https://gw.example/api/messages")):
            self.assertEqual(custom_endpoint(given, api_format), expected)
        for bad in ("soap", None, ["chat"]):
            with self.assertRaises(UserError): custom_endpoint("https://gw.example/v1", bad)
        with self.assertRaises(UserError): custom_endpoint("https://gw.example/v1/chat/completions?a=1", "responses")
        for bad in ("", None, "api.deepseek.com/v1", "file:///etc/passwd", "https://", "https://a b.example/v1", "https://中文.example/v1",
                    "https://key@x.example/v1", "https://x.example/v1#part", "https://x.example/v1?a=1", "https://x.example:99999/v1", "x" * 600):
            with self.assertRaises(UserError): custom_endpoint(bad)

    def test_file_paths_and_clip_ranges_are_validated(self):
        with self.assertRaises(UserError): probe_media("https://example.com/movie.mp4")
        with self.assertRaises(UserError): probe_media(r"\\server\share\movie.mp4")
        media = Media("x", Path("test.mp4"), 100, 1, 0, "video")
        self.assertEqual(clip_range(media, 90, 30), (90, 10))
        for start, duration in ((-1, 2), (100, 1), (0, float("nan")), (float("inf"), 1)):
            with self.assertRaises(UserError): clip_range(media, start, duration)


class Reply:
    def __init__(self, payload, headers=None): self.payload, self.headers = payload, headers or {}
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def read(self, limit): return json.dumps(self.payload).encode()


class Stream(Reply):
    """A server-sent event stream made of the given events."""
    def __init__(self, events): super().__init__(events, {"Content-Type": "text/event-stream; charset=utf-8"})
    def read(self, limit):
        return "".join(f"event: {e.get('type')}\ndata: {json.dumps(e, ensure_ascii=False)}\n\n" for e in self.payload).encode() + b"data: [DONE]\n\n"


def http_error(code, message):
    return urllib.error.HTTPError("https://llm.example", code, "error", {}, BytesIO(json.dumps({"error": {"message": message}}).encode()))


class CustomClientTests(unittest.TestCase):
    CUES = [{"id": 1, "text": "こんにちは"}]
    ANSWER = {"choices": [{"finish_reason": "stop", "message": {"content": '{"translations":[{"id":1,"text":"你好"}]}'}}], "usage": None}

    def call(self, replies, key="custom-key", client=None, endpoint="https://llm.example/v1/chat/completions", api_format="chat"):
        client = client or CustomClient(endpoint, api_format=api_format)
        with patch("providers.urllib.request.build_opener") as build:
            build.return_value.open.side_effect = replies
            try:
                result = client.translate(self.CUES, "zh-CN", key, source_language="ja", model="my-model", context_hint="访谈")
            except Exception as exc:
                result = exc
            sent = [call.args[0] for call in build.return_value.open.call_args_list]
            handlers = [type(h).__name__ for h in build.call_args.args]
        return result, sent, handlers, client

    def test_request_goes_to_the_given_address_with_standard_parameters(self):
        (mapping, usage, _), sent, handlers, _ = self.call([Reply(self.ANSWER)])
        self.assertEqual((mapping, usage), ({1: "你好"}, {}))
        self.assertEqual((sent[0].full_url, sent[0].get_header("Authorization")), ("https://llm.example/v1/chat/completions", "Bearer custom-key"))
        body = json.loads(sent[0].data)
        self.assertEqual({k: v for k, v in body.items() if k != "messages"},
                         {"model": "my-model", "stream": False, "temperature": 0.1, "response_format": {"type": "json_object"}})
        self.assertIn("访谈", body["messages"][0]["content"])
        # Redirects are never followed with the key; a public address may use the system proxy.
        self.assertEqual(handlers, ["NoRedirect"])

    def test_no_key_means_no_authorization_header_and_local_servers_skip_the_proxy(self):
        (mapping, _, _), sent, handlers, _ = self.call([Reply(self.ANSWER)], key="", endpoint="http://localhost:11434/v1/chat/completions")
        self.assertEqual(mapping, {1: "你好"})
        self.assertIsNone(sent[0].get_header("Authorization"))
        self.assertEqual(handlers, ["NoRedirect", "ProxyHandler"])

    def test_optional_parameters_are_dropped_once_a_server_turns_them_down(self):
        (mapping, _, _), sent, _, client = self.call([http_error(400, "response_format is not supported"), Reply(self.ANSWER)])
        self.assertEqual((mapping, len(sent), client.plain), ({1: "你好"}, 2, True))
        self.assertEqual(set(json.loads(sent[1].data)), {"model", "stream", "messages"})
        _, sent, _, _ = self.call([Reply(self.ANSWER)], client=client)
        self.assertEqual((len(sent), set(json.loads(sent[0].data))), (1, {"model", "stream", "messages"}))

    def test_errors_that_are_not_about_parameters_are_not_sent_again(self):
        for code, expected in ((401, "请更新 API Key"), (404, "/v1"), (429, "稍后"), (500, "HTTP 500"), (301, "最终地址")):
            error, sent, _, client = self.call([http_error(code, "nope custom-key")])
            self.assertIsInstance(error, UserError)
            self.assertNotIsInstance(error, ContentError)
            self.assertIn(expected, str(error))
            self.assertNotIn("custom-key", str(error))
            self.assertEqual((len(sent), client.plain), (1, False))
        refused, sent, _, _ = self.call([http_error(451, "prohibited or sensitive content")])
        self.assertIsInstance(refused, ContentError)
        flagged, sent, _, _ = self.call([http_error(400, "Content Exists Risk")])
        self.assertEqual((type(flagged), len(sent)), (ContentError, 1))
        both, sent, _, client = self.call([http_error(400, "bad request"), http_error(400, "model not found")])
        self.assertIn("model not found", str(both))
        self.assertEqual((len(sent), client.plain), (2, False))
        with self.assertRaises(UserError):
            CustomClient("file:///C:/secret")
        with self.assertRaises(UserError):
            CustomClient("https://llm.example/v1/chat/completions", api_format="soap")

    JSON_TEXT = '{"translations":[{"id":1,"text":"你好"}]}'

    def responses(self, replies, **options):
        return self.call(replies, endpoint="https://gw.example/v1/responses", api_format="responses", **options)

    def completed(self, status="completed", content=None, **more):
        content = [{"type": "output_text", "text": self.JSON_TEXT}] if content is None else content
        return {"id": "resp_1", "status": status, "error": None, "usage": {"input_tokens": 30, "output_tokens": 7, "total_tokens": 37},
                "output": [{"type": "reasoning", "summary": []}, {"type": "message", "role": "assistant", "content": content}], **more}

    def test_responses_format_streams_and_reads_the_final_answer(self):
        events = [{"type": "response.created", "response": {"status": "in_progress"}},
                  {"type": "response.output_text.delta", "delta": self.JSON_TEXT[:10]},
                  {"type": "response.output_text.delta", "delta": self.JSON_TEXT[10:]},
                  {"type": "response.completed", "response": self.completed()}]
        (mapping, usage, _), sent, _, _ = self.responses([Stream(events)])
        self.assertEqual((mapping, usage), ({1: "你好"}, {"prompt_tokens": 30, "completion_tokens": 7}))
        self.assertEqual((sent[0].full_url, sent[0].get_header("Authorization"), sent[0].get_header("X-api-key")),
                         ("https://gw.example/v1/responses", "Bearer custom-key", None))
        body = json.loads(sent[0].data)
        self.assertEqual({k: v for k, v in body.items() if k not in {"instructions", "input"}},
                         {"model": "my-model", "stream": True, "store": False, "text": {"format": {"type": "json_object"}}})
        self.assertIn("访谈", body["instructions"])
        text = body["input"][0]["content"][0]
        self.assertEqual((body["input"][0]["role"], text["type"]), ("user", "input_text"))
        # JSON mode is only accepted when the input itself says JSON.
        self.assertIn("JSON", text["text"])
        self.assertIn("こんにちは", text["text"])
        # A server that answers in one piece, or whose final event carries no output, is read as well.
        (mapping, _, _), _, _, _ = self.responses([Reply(self.completed())])
        self.assertEqual(mapping, {1: "你好"})
        (mapping, _, _), _, _, _ = self.responses([Stream([events[1], events[2], {"type": "response.completed", "response": {"status": "completed", "output": []}}])])
        self.assertEqual(mapping, {1: "你好"})

    def test_responses_format_drops_json_mode_when_it_is_turned_down(self):
        done = Stream([{"type": "response.completed", "response": self.completed()}])
        (mapping, _, _), sent, _, client = self.responses([http_error(400, "Unsupported parameter: text.format"), done])
        self.assertEqual((mapping, len(sent), client.plain), ({1: "你好"}, 2, True))
        self.assertEqual(set(json.loads(sent[1].data)), {"model", "instructions", "input", "stream", "store"})

    def test_responses_format_never_takes_part_of_an_answer(self):
        delta = {"type": "response.output_text.delta", "delta": self.JSON_TEXT}
        cut, sent, _, _ = self.responses([Stream([delta])])
        # The connection ended before the answer did: maybe billed, so it is not sent again by itself.
        self.assertEqual((type(cut), len(sent)), (UserError, 1))
        self.assertIn("没有传完", str(cut))
        for events in ([delta, {"type": "response.incomplete", "response": self.completed("incomplete", incomplete_details={"reason": "max_output_tokens"})}],
                       [{"type": "response.completed", "response": self.completed(content=[{"type": "refusal", "refusal": "no"}])}],
                       [{"type": "response.completed", "response": self.completed(content=[])}],
                       [{"type": "response.failed", "response": {"status": "failed", "error": {"code": "x", "message": "flagged by content policy"}}}]):
            self.assertIsInstance(self.responses([Stream(events)])[0], ContentError, events)
        for events in ([{"type": "response.failed", "response": {"status": "failed", "error": {"code": "server_error", "message": "upstream broke custom-key"}}}],
                       [delta, {"type": "error", "code": "rate_limit_exceeded", "message": "slow down"}],
                       [{"type": "error", "error": {"type": "overloaded_error", "message": "busy"}}]):
            failed = self.responses([Stream(events)])[0]
            self.assertEqual((type(failed), "custom-key" in str(failed)), (UserError, False), events)

    MESSAGE = {"type": "message", "role": "assistant", "stop_reason": "end_turn", "usage": {"input_tokens": 12, "output_tokens": 5},
               "content": [{"type": "thinking", "thinking": "…"}, {"type": "text", "text": JSON_TEXT}]}

    def messages(self, replies, **options):
        return self.call(replies, endpoint="https://gw.example/v1/messages", api_format="messages", **options)

    def test_messages_format_speaks_anthropic(self):
        (mapping, usage, _), sent, _, _ = self.messages([Reply(self.MESSAGE)])
        self.assertEqual((mapping, usage), ({1: "你好"}, {"prompt_tokens": 12, "completion_tokens": 5}))
        self.assertEqual((sent[0].full_url, sent[0].get_header("X-api-key"), sent[0].get_header("Anthropic-version"), sent[0].get_header("Authorization")),
                         ("https://gw.example/v1/messages", "custom-key", "2023-06-01", None))
        body = json.loads(sent[0].data)
        self.assertEqual({k: v for k, v in body.items() if k not in {"system", "messages"}}, {"model": "my-model", "max_tokens": 8192, "temperature": 0.1})
        self.assertIn("访谈", body["system"])
        self.assertEqual((len(body["messages"]), body["messages"][0]["role"]), (1, "user"))
        for reason in ("max_tokens", "refusal"):
            self.assertIsInstance(self.messages([Reply({**self.MESSAGE, "stop_reason": reason})])[0], ContentError)
        self.assertIsInstance(self.messages([Reply({**self.MESSAGE, "content": []})])[0], ContentError)
        wrong = self.messages([Reply(self.ANSWER)])[0]
        self.assertEqual((type(wrong), "格式错误" in str(wrong)), (UserError, True))
        # Without the optional parameters after a 400, as in the other formats.
        (mapping, _, _), sent, _, client = self.messages([http_error(400, "temperature is not supported"), Reply(self.MESSAGE)])
        self.assertEqual((mapping, client.plain, json.loads(sent[1].data).get("temperature"), json.loads(sent[1].data)["max_tokens"]),
                         ({1: "你好"}, True, None, 4096))

    def test_messages_format_falls_back_to_a_bearer_key_after_401(self):
        (mapping, _, _), sent, _, client = self.messages([http_error(401, "missing bearer token"), Reply(self.MESSAGE)])
        self.assertEqual((mapping, len(sent), client.auth), ({1: "你好"}, 2, "bearer"))
        self.assertEqual((sent[1].get_header("Authorization"), sent[1].get_header("X-api-key"), sent[1].get_header("Anthropic-version")),
                         ("Bearer custom-key", None, "2023-06-01"))
        _, sent, _, _ = self.call([Reply(self.MESSAGE)], client=client)
        self.assertEqual((len(sent), sent[0].get_header("Authorization")), (1, "Bearer custom-key"))
        # A key that is simply wrong is reported after both forms were tried; without a key there is nothing to try.
        wrong, sent, _, _ = self.messages([http_error(401, "invalid key"), http_error(401, "invalid key")])
        self.assertEqual((type(wrong).__name__, len(sent), "请更新 API Key" in str(wrong)), ("ProviderError", 2, True))
        _, sent, _, _ = self.messages([http_error(401, "key needed")], key="")
        self.assertEqual(len(sent), 1)

    def test_error_naming_another_endpoint_points_at_the_format_setting(self):
        gateway = "/v1/chat/completions serves the DeepSeek models only; use /v1/messages for Claude models and /v1/responses for GPT models"
        error, sent, _, _ = self.call([http_error(400, gateway), http_error(400, gateway)])
        self.assertEqual(len(sent), 2)
        # The model of the test (my-model) belongs to neither family named, so both formats are suggested.
        self.assertIn("请把“接口格式”改为 Responses 或 Anthropic Messages", str(error))
        self.assertIn("use /v1/messages", str(error))
        for model, expected in (("gpt-6.1-sol", "改为 Responses 后"), ("claude-sonnet-5-5", "改为 Anthropic Messages 后")):
            client = CustomClient("https://gw.example/v1/chat/completions")
            with patch("providers.urllib.request.build_opener") as build:
                build.return_value.open.side_effect = [http_error(400, gateway), http_error(400, gateway)]
                with self.assertRaises(UserError) as raised:
                    client.translate(self.CUES, "zh-CN", "custom-key", model=model)
            self.assertIn(expected, str(raised.exception))
        # An address that does not exist on the server may be the wrong format, too.
        missing = self.responses([http_error(404, "Not Found")])[0]
        self.assertIn("接口格式", str(missing))
        plain = self.call([http_error(404, "model not found")])[0]
        self.assertIn("接口格式", str(plain))
        self.assertNotIn("接口格式", str(self.call([http_error(429, "slow down")])[0]))


class SpeechDetectorTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.wav = Path(self.temp.name) / "speech.wav"
        rng = np.random.default_rng(7)
        # 20.02 s (not a whole number of frames): noise bursts between silences.
        count = 16000 * 20 + 300
        envelope = (np.sin(np.arange(count) / 16000 * 1.3) > 0.2).astype(np.float32)
        samples = (rng.standard_normal(count) * 6000 * envelope).astype("<i2")
        with wave.open(str(self.wav), "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(samples.tobytes())
        self.samples = samples

    def tearDown(self): self.temp.cleanup()

    def reference(self):
        """The detector as it was first written: one frame per model call, straight from the samples."""
        import onnxruntime as ort
        from media import VAD_PATH
        session = ort.InferenceSession(str(VAD_PATH), providers=["CPUExecutionProvider"])
        state, context, probabilities = np.zeros((2, 1, 128), dtype=np.float32), np.zeros((1, 64), dtype=np.float32), []
        for first in range(0, len(self.samples), 512):
            frame = self.samples[first:first + 512].astype(np.float32) / 32768.0
            frame = np.pad(frame, (0, 512 - len(frame)))
            chunk = np.concatenate((context, frame.reshape(1, -1)), axis=1)
            output, state = session.run(None, {"input": chunk, "state": state, "sr": np.array(16000, dtype=np.int64)})
            probabilities.append(float(output.reshape(-1)[0]))
            context = chunk[:, -64:]
        return probabilities

    def probabilities(self, detector):
        """What the stream pass alone gives: the second pass is replaced by one that finds nothing."""
        seen = {}
        def capture(probabilities, duration, **options):
            seen["probabilities"], seen["duration"] = probabilities, duration
            return []
        progress = []
        detector.windows = lambda session, wav, frames, cancel, report: np.zeros(frames, dtype=np.float32)
        with patch("media.probabilities_to_segments", capture):
            detector.detect(self.wav, lambda ratio, label: progress.append(ratio), Event())
        self.assertEqual(progress[-1], 1)
        self.assertTrue(all(0 < ratio <= 1 for ratio in progress))
        return seen["probabilities"], seen["duration"]

    def test_one_stream_gives_exactly_the_frame_by_frame_result(self):
        detector = SpeechDetector(workers=4, firered=False)
        detector.BLOCK = 7  # several blocks, the last one short
        probabilities, duration = self.probabilities(detector)
        self.assertEqual(probabilities, self.reference())
        self.assertAlmostEqual(duration, 20.01875)

    def test_long_files_are_split_and_each_part_replays_what_came_before(self):
        reference = self.reference()
        detector = SpeechDetector(workers=3, firered=False)
        detector.PART_SECONDS, detector.WARMUP_SECONDS = 5, 2
        probabilities, _ = self.probabilities(detector)
        size = -(-len(reference) // 3)
        self.assertEqual(len(probabilities), len(reference))
        # The first part starts like the single stream; later parts start from a short replay and only come close.
        self.assertEqual(probabilities[:size], reference[:size])
        self.assertNotEqual(probabilities, reference)
        whole = SpeechDetector(workers=3, firered=False)
        whole.PART_SECONDS, whole.WARMUP_SECONDS = 5, 1000
        self.assertEqual(self.probabilities(whole)[0], reference)
        self.assertEqual(self.probabilities(SpeechDetector(workers=1, firered=False))[0], reference)

    def test_cancel_stops_every_part(self):
        detector = SpeechDetector(workers=3)
        detector.PART_SECONDS, detector.BLOCK = 5, 10
        cancel, calls = Event(), []
        def progress(ratio, label):
            calls.append(ratio)
            cancel.set()
        with self.assertRaises(Cancelled):
            detector.detect(self.wav, progress, cancel)
        self.assertLess(len(calls), 12)

    def fresh(self, first, last):
        """Frame-by-frame probabilities of frames first..last from an empty state."""
        import onnxruntime as ort
        from media import VAD_PATH
        session = ort.InferenceSession(str(VAD_PATH), providers=["CPUExecutionProvider"])
        padded = np.concatenate((np.zeros(64, dtype=np.float32), self.samples.astype(np.float32) / 32768.0, np.zeros(512, dtype=np.float32)))
        state, probabilities = np.zeros((2, 1, 128), dtype=np.float32), []
        for index in range(first, last):
            output, state = session.run(None, {"input": padded[index * 512:index * 512 + 576].reshape(1, -1), "state": state,
                                               "sr": np.array(16000, dtype=np.int64)})
            probabilities.append(float(output.reshape(-1)[0]))
        return probabilities

    def test_second_pass_hears_overlapping_windows_that_each_start_empty(self):
        detector = SpeechDetector(workers=3)
        detector.WINDOW_SECONDS = 4  # 125 frames, a new window every 62
        frames = -(-len(self.samples) // 512)
        reports = []
        probabilities = detector.windows(detector.load(), self.wav, frames, Event(), reports.append)
        expected = np.zeros(frames, dtype=np.float32)
        for first in range(0, frames, 62):
            last = min(frames, first + 125)
            expected[first:last] = np.maximum(expected[first:last], np.array(self.fresh(first, last), dtype=np.float32))
        self.assertEqual(probabilities.tolist(), expected.tolist())
        self.assertAlmostEqual(max(reports), 1)
        # A window that starts later differs from the stream: the state is what the second pass leaves behind.
        self.assertNotEqual(probabilities.tolist(), self.reference())

    def test_either_pass_makes_a_sentence_and_those_of_the_second_alone_are_marked(self):
        detector = SpeechDetector(workers=1, firered=False)
        frames = -(-len(self.samples) // 512)
        stream, fresh = np.zeros(frames, dtype=np.float32), np.zeros(frames, dtype=np.float32)
        stream[100:150] = 0.9  # 3.2 to 4.8 s
        fresh[95:160] = 0.8  # the same sentence, heard a little longer
        fresh[300:340] = 0.7  # 9.6 to 10.9 s: only the second pass hears it
        fresh[500:520] = 0.4  # below the threshold
        detector.analyse = lambda session, wav, spans, cancel, report: [stream]
        detector.windows = lambda session, wav, frames, cancel, report: fresh
        labels = []
        segments = detector.detect(self.wav, lambda ratio, label: labels.append(label), Event())
        self.assertEqual(segments, [{"start": 2.96, "end": 5.22}, {"start": 9.52, "end": 10.98, "recovered": True}])
        self.assertIn("2 句，其中 1 句是复查时才找到的", labels[-1])

    def test_all_passes_run_on_real_audio(self):
        progress = []
        segments = SpeechDetector(workers=2).detect(self.wav, lambda ratio, label: progress.append((ratio, label)), Event())
        self.assertEqual(progress[-1][0], 1)
        self.assertTrue(all(0 < ratio <= 1 for ratio, _ in progress))
        self.assertTrue(all(a <= b for (a, _), (b, _) in zip(progress, progress[1:])))
        self.assertTrue(any("复查" in label for _, label in progress[:-1]))
        self.assertTrue(any("FireRedVAD" in label for _, label in progress[:-1]))
        # The test audio is bursts of white noise: Silero takes some for speech, FireRedVAD does not.
        self.assertEqual(segments, [])
        self.assertTrue(all(a["end"] <= b["start"] for a, b in zip(segments, segments[1:])))

    def test_firered_cuts_the_sentences_and_silero_says_how_sure_they_are(self):
        detector = SpeechDetector(workers=1)
        frames = -(-len(self.samples) // 512)
        stream, fresh = np.zeros(frames, dtype=np.float32), np.zeros(frames, dtype=np.float32)
        stream[100:150] = 0.9  # 3.2 to 4.8 s
        fresh[300:340] = 0.7  # 9.6 to 10.9 s: only Silero's second pass
        stream[560:580] = 0.9  # 17.9 to 18.6 s: Silero only, FireRedVAD hears nothing
        heard = np.zeros(-(-len(self.samples) // 160), dtype=np.float32)
        heard[300:500] = 0.95  # 3 to 5 s
        heard[960:1090] = 0.9  # 9.6 to 10.9 s
        heard[1500:1560] = 0.8  # 15 to 15.6 s: FireRedVAD alone
        calls = []
        class Fake:
            def available(self): return True
            def probabilities(self, wav, workers, cancel, report):
                calls.append(workers)
                report(0.5)
                report(1.0)
                return heard
        detector.firered = Fake()
        detector.analyse = lambda session, wav, spans, cancel, report: [stream]
        detector.windows = lambda session, wav, frames, cancel, report: fresh
        progress = []
        segments = detector.detect(self.wav, lambda ratio, label: progress.append((ratio, label)), Event())
        self.assertEqual(calls, [1])
        self.assertEqual([(round(s["start"], 2), round(s["end"], 2), s.get("recovered", False)) for s in segments],
                         [(2.9, 5.09, False), (9.52, 11.01, False), (14.9, 15.72, True)])
        self.assertEqual(progress[-2][0], 1)
        self.assertIn("用 FireRedVAD 再听一遍", progress[-3][1])
        self.assertIn("3 句，其中 1 句是复查时才找到的", progress[-1][1])

    def test_without_firered_weights_silero_alone_decides(self):
        from firered_vad import FireRedVad
        detector = SpeechDetector(workers=1, firered=FireRedVad(Path(self.temp.name) / "missing.npz"))
        silero = SpeechDetector(workers=1, firered=False)
        labels = []
        self.assertEqual(detector.detect(self.wav, lambda ratio, label: labels.append(label), Event()),
                         silero.detect(self.wav, lambda ratio, label: None, Event()))
        self.assertFalse(any("FireRedVAD" in label for label in labels))


class FireRedVadTests(unittest.TestCase):
    """The NumPy port against values of the original code (kaldi-native-fbank 1.23 and FireRedVAD 0.0.2 on PyTorch)."""

    @staticmethod
    def signal(seconds):
        t = np.arange(int(16000 * seconds)) / 16000
        voice = sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 12)) * (0.5 + 0.5 * np.sin(2 * np.pi * 3 * t))
        gate = np.sin(2 * np.pi * 0.4 * t) > -0.3
        return (4000 * voice * gate + 300 * np.sin(2 * np.pi * 2900 * t * t)).astype("<i2")

    def test_filterbank_matches_kaldi(self):
        from firered_vad import kaldi_fbank
        features = kaldi_fbank(self.signal(4.0))
        self.assertEqual(features.shape, (398, 80))
        expected = [[13.7958, 18.9593, 9.288, 5.6993], [9.9996, 15.6973, 6.8539, 6.789],
                    [11.9732, 18.0012, 8.402, 5.9539], [0.8482, -0.2038, 2.6713, 7.3555]]
        np.testing.assert_allclose(features[np.ix_([0, 120, 250, 397], [0, 9, 40, 79])], expected, atol=2e-3)
        self.assertEqual(kaldi_fbank(np.zeros(399, dtype="<i2")).shape, (0, 80))
        self.assertEqual(kaldi_fbank(np.zeros(400, dtype="<i2")).shape, (1, 80))

    def test_network_matches_the_original(self):
        from firered_vad import FireRedVad, kaldi_fbank
        probabilities = FireRedVad().forward(kaldi_fbank(self.signal(4.0)))
        expected = [0.35384, 0.87098, 0.71122, 0.00133, 0.00137, 0.4978, 0.01303]
        np.testing.assert_allclose(probabilities[[0, 40, 100, 160, 220, 300, 397]], expected, atol=1e-4)

    def test_long_files_are_heard_in_independent_pieces_on_several_threads(self):
        import firered_vad
        from firered_vad import FireRedVad, kaldi_fbank
        samples = self.signal(10.0)
        with TemporaryDirectory() as folder:
            path = Path(folder) / "audio.wav"
            with wave.open(str(path), "wb") as target:
                target.setnchannels(1); target.setsampwidth(2); target.setframerate(16000)
                target.writeframes(samples.tobytes())
            model = FireRedVad()
            features = kaldi_fbank(samples)
            with patch.object(firered_vad, "CHUNK_FRAMES", 300):
                expected = np.concatenate([model.forward(features[first:first + 300]) for first in range(0, len(features), 300)])
                reports = []
                for workers in (1, 3):
                    got = model.probabilities(path, workers, Event(), reports.append)
                    np.testing.assert_allclose(got, expected, atol=1e-6)
                self.assertEqual(max(reports), 1)
                # Pieces are heard without each other, so the edges differ from one whole pass.
                self.assertGreater(np.abs(expected - model.forward(features)).max(), 1e-3)
                cancel = Event()
                cancel.set()
                with self.assertRaises(Cancelled):
                    model.probabilities(path, 2, cancel, reports.append)

    def test_probabilities_go_onto_the_coarser_grid_as_the_highest_value(self):
        from firered_vad import to_grid
        # 10 ms frames onto 32 ms frames: frame i starts at i * 160 samples, inside grid frame i * 160 // 512.
        grid = to_grid(np.array([0.1, 0.9, 0.2, 0.3, 0.8, 0.4, 0.5], dtype=np.float32), 2, 512)
        self.assertEqual(grid.tolist(), [np.float32(0.9), np.float32(0.8)])
        self.assertEqual(to_grid(np.zeros(0, dtype=np.float32), 3, 512).tolist(), [0, 0, 0])

    def test_damaged_weights_are_reported(self):
        from firered_vad import FireRedVad
        with TemporaryDirectory() as folder:
            path = Path(folder) / "fireredvad.npz"
            path.write_bytes(b"not a model")
            with self.assertRaises(UserError):
                FireRedVad(path).load()


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

    def test_waits_for_the_graphics_card_and_releases_it(self):
        from local_asr import GpuGate, LocalAsrEngine
        from threading import Timer
        gate = GpuGate()
        gate.acquire("去马赛克", Event())
        engine = LocalAsrEngine("Fake", python=self.engine.python, worker=self.engine.worker, models=self.engine.models,
                                log_dir=self.engine.log_dir, gate=gate)
        stages, cancel = [], Event()
        Timer(0.3, cancel.set).start()
        with self.assertRaises(Cancelled):
            engine.start(cancel, stages.append)
        self.assertIsNone(engine.process)
        self.assertIn("去马赛克正在使用显卡", stages[0])
        Timer(0.3, gate.release).start()
        engine.start(Event(), stages.append)
        self.assertEqual(gate.holder, "本地语音识别")
        engine.close()
        self.assertFalse(gate.busy())

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
