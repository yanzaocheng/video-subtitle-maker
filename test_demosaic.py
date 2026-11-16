from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import os
import subprocess
import sys
import time
import unittest

import imageio_ffmpeg

from common import Cancelled, UserError
from demosaic import MosaicRemover, parse_progress, progress_label, restored_name
from local_asr import GpuGate
from media import probe_media

# Stands in for lada-cli.exe: same arguments, a tqdm-style progress bar on stderr, then the output file.
FAKE_LADA = r'''
import argparse, os, shutil, sys, time
parser = argparse.ArgumentParser()
for name in ("--input", "--output", "--temporary-directory", "--encoding-preset"):
    parser.add_argument(name)
parser.add_argument("--detect-face-mosaics", action="store_true")
args = parser.parse_args()
mode = os.environ.get("FAKE_LADA_MODE", "ok")
assert os.environ["LANGUAGE"] == "en" and os.path.isdir(args.temporary_directory)
for done in (0, 50, 100):
    sys.stderr.write("\rProcessing video: %3d%%|##|Processed: 0:01 (%df) | Remaining: 0:01 (%df) | Speed: 25.0fps" % (done, done, 100 - done))
    sys.stderr.flush()
    time.sleep(0.05)
if mode == "hang":
    time.sleep(60)
if mode == "crash":
    print("Traceback: boom", flush=True)
    sys.exit(3)
if mode == "silent":
    print("Error on export: frame restorer stopped prematurely", flush=True)
    sys.exit(0)
sys.stderr.write("\n")
print("Processing audio", flush=True)
shutil.copyfile(args.input, args.output)
'''


class ProgressParsingTests(unittest.TestCase):
    def test_reads_lada_progress_bar(self):
        line = "Processing video:  45%|████▌     |Processed: 01:23 (2000f) | Remaining: 1:40:05 (2400f) | Speed: 24.0fps"
        info = parse_progress(line)
        self.assertEqual(info, {"percent": 45, "frames_done": 2000, "frames_total": 4400, "fps": 24.0, "remaining": "1:40:05"})
        self.assertEqual(progress_label(info), "去马赛克：2000 / 4400 帧 · 24.0 帧/秒 · 预计还需 1:40:05")
        early = parse_progress("Processing video:   0%|          |Processed: 00:00 (0f) | Remaining: ? | Speed: ?")
        self.assertEqual((early["frames_done"], early["frames_total"], early["fps"]), (0, 0, 0.0))
        for other in ("Processing audio", "Error on export 100%", ""):
            self.assertIsNone(parse_progress(other))

    def test_output_names(self):
        self.assertEqual(restored_name(Path("C:/v/片名.mkv")), "片名.restored.mp4")
        self.assertEqual(restored_name(Path("C:/v/a.b.mp4"), 3725.9), "a.b.restored.sample-1h02m05s.mp4")


class GpuGateTests(unittest.TestCase):
    def test_second_task_waits_and_can_cancel(self):
        gate, waits = GpuGate(), []
        gate.acquire("去马赛克", Event())
        cancel = Event()
        Thread(target=lambda: (time.sleep(0.3), cancel.set()), daemon=True).start()
        with self.assertRaises(Cancelled):
            gate.acquire("本地语音识别", cancel, waits.append)
        self.assertEqual(waits, ["去马赛克"])
        Thread(target=lambda: (time.sleep(0.3), gate.release()), daemon=True).start()
        gate.acquire("本地语音识别", Event())
        self.assertEqual(gate.holder, "本地语音识别")
        gate.release()
        self.assertFalse(gate.busy())


class MosaicRemoverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.video = root / "source" / "movie.mp4"
        cls.video.parent.mkdir()
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=10:d=4",
                   "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=4", "-c:v", "mpeg4", "-g", "10", "-c:a", "aac",
                   "-shortest", str(cls.video)]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        cls.cli = root / "lada" / "fake_lada.py"
        cls.cli.parent.mkdir()
        cls.cli.write_text(FAKE_LADA, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.gate = GpuGate()
        self.remover = MosaicRemover(cli=self.cli, command=[sys.executable, str(self.cli)], gate=self.gate,
                                     log_dir=Path(self.temp.name) / "logs")
        self.media = probe_media(self.video)
        self.updates = []
        os.environ["FAKE_LADA_MODE"] = "ok"

    def tearDown(self):
        os.environ.pop("FAKE_LADA_MODE", None)
        for path in self.video.parent.iterdir():
            if path != self.video:
                path.unlink()
        self.assertFalse(self.gate.busy())

    def update(self, **changes):
        self.updates.append(changes)

    def leftovers(self):
        return [p.name for p in self.video.parent.iterdir() if p.name.startswith(".demosaic-")]

    def test_whole_video_is_saved_next_to_the_source(self):
        output = self.remover.run(self.media, None, self.update, Event())
        self.assertEqual(output, self.video.parent / "movie.restored.mp4")
        self.assertEqual(output.read_bytes(), self.video.read_bytes())
        stages = [u["stage"] for u in self.updates if "stage" in u]
        self.assertIn("去马赛克：50 / 100 帧 · 25.0 帧/秒 · 预计还需 0:01", stages)
        self.assertIn("正在合并原音轨", stages)
        self.assertEqual(self.leftovers(), [])
        # A second run never replaces the first result.
        self.assertEqual(self.remover.run(self.media, None, self.update, Event()).name, "movie.restored.1.mp4")

    def test_sample_cuts_a_clip_first(self):
        output = self.remover.run(self.media, 1.0, self.update, Event())
        self.assertEqual(output.name, "movie.restored.sample-0h00m01s.mp4")
        clip = probe_media(output)
        self.assertLess(clip.duration, self.media.duration)
        self.assertTrue(any("截取试处理片段" in u.get("stage", "") for u in self.updates))

    def test_failures_leave_no_output(self):
        for mode, message in (("silent", "Error on export"), ("crash", "退出码 3")):
            os.environ["FAKE_LADA_MODE"] = mode
            with self.assertRaises(UserError) as caught:
                self.remover.run(self.media, None, self.update, Event())
            self.assertIn(message, str(caught.exception))
            self.assertEqual(sorted(p.name for p in self.video.parent.iterdir()), ["movie.mp4"])

    def test_cancel_stops_the_program(self):
        os.environ["FAKE_LADA_MODE"] = "hang"
        cancel = Event()
        Thread(target=lambda: (time.sleep(1.5), cancel.set()), daemon=True).start()
        started = time.monotonic()
        with self.assertRaises(Cancelled):
            self.remover.run(self.media, None, self.update, cancel)
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(sorted(p.name for p in self.video.parent.iterdir()), ["movie.mp4"])

    def test_missing_program_and_changed_video_are_reported(self):
        missing = MosaicRemover(cli=Path(self.temp.name) / "none.exe", gate=self.gate)
        with self.assertRaises(UserError) as caught:
            missing.run(self.media, None, self.update, Event())
        self.assertIn("setup_lada.py", str(caught.exception))
        os.utime(self.video, ns=(self.media.modified_ns + 10 ** 9, self.media.modified_ns + 10 ** 9))
        try:
            with self.assertRaises(UserError):
                self.remover.run(self.media, None, self.update, Event())
        finally:
            os.utime(self.video, ns=(self.media.modified_ns, self.media.modified_ns))


if __name__ == "__main__":
    unittest.main()
