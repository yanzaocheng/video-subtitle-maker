"""FireRedVAD speech detection in NumPy.

FireRedVAD (https://github.com/FireRedTeam/FireRedVAD, Apache-2.0, see assets/fireredvad-LICENSE.txt) is a small
DFSMN network over Kaldi filterbank features. assets/fireredvad.npz holds its released "VAD" weights and CMVN
statistics unchanged, converted from model.pth.tar / cmvn.ark to plain arrays, so neither PyTorch nor Kaldi is needed.
Against the original PyTorch code the probabilities differ by less than 1e-4.

It hears quiet speech and speech under noise (water, music) that Silero misses; it also takes more moans for speech.
"""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
import wave

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from common import UserError, check_cancel

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "fireredvad.npz"
SAMPLE_RATE = 16000
WINDOW, SHIFT = 400, 160  # 25 ms frames every 10 ms
# The original splits long input into pieces of this many frames, each heard without the others.
CHUNK_FRAMES = 30000
TAPS = 20  # memory of each FSMN layer: 20 frames back (up to the current one) and 20 ahead
BLOCKS = 7


def _mel(frequency):
    return 1127.0 * np.log(1.0 + frequency / 700.0)


def _mel_banks(bins=80, fft=512, low=20.0):
    """Kaldi's triangular mel filters (high frequency = Nyquist), over the first fft/2 + 1 power bins."""
    first, last = _mel(low), _mel(SAMPLE_RATE / 2)
    delta = (last - first) / (bins + 1)
    mels = _mel(np.arange(fft // 2) * SAMPLE_RATE / fft)
    banks = np.zeros((bins, fft // 2 + 1))
    for index in range(bins):
        left, centre, right = first + index * delta, first + (index + 1) * delta, first + (index + 2) * delta
        rising, falling = (mels > left) & (mels <= centre), (mels > centre) & (mels < right)
        banks[index, :fft // 2][rising] = (mels[rising] - left) / (centre - left)
        banks[index, :fft // 2][falling] = (right - mels[falling]) / (right - centre)
    return banks


_BANKS = _mel_banks()
_POVEY = np.power(0.5 - 0.5 * np.cos(2 * np.pi * np.arange(WINDOW) / (WINDOW - 1)), 0.85)


def kaldi_fbank(samples):
    """80 log mel energies per 10 ms of int16 samples, as Kaldi computes them (no dither, frames inside the signal)."""
    count = 1 + (len(samples) - WINDOW) // SHIFT if len(samples) >= WINDOW else 0
    if count == 0:
        return np.zeros((0, _BANKS.shape[0]), dtype=np.float32)
    frames = sliding_window_view(np.asarray(samples, dtype=np.float64), WINDOW)[::SHIFT][:count].copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] *= 0.03
    frames *= _POVEY
    energies = (np.abs(np.fft.rfft(frames, n=512)) ** 2) @ _BANKS.T
    return np.log(np.maximum(energies, np.finfo(np.float32).eps)).astype(np.float32)


class FireRedVad:
    def __init__(self, path=MODEL_PATH):
        self.path = Path(path)
        self.weights, self.lock = None, Lock()

    def available(self):
        return self.path.is_file()

    def load(self):
        with self.lock:
            if self.weights is None:
                try:
                    with np.load(self.path) as data:
                        self.weights = {name: data[name] for name in data.files}
                except (OSError, ValueError):
                    raise UserError("语音检测模型 assets/fireredvad.npz 缺失或已损坏，请重新下载程序。") from None
            return self.weights

    def forward(self, features):
        """Speech probability of each feature frame."""
        w = self.load()

        def dense(x, name, bias=True, relu=True):
            y = x @ w[name + ".weight"].T
            if bias:
                y += w[name + ".bias"]
            return np.maximum(y, 0, out=y) if relu else y

        def fsmn(x, name):
            # memory = x + taps over x[t-19..t] (lookback) + taps over x[t+1..t+20] (lookahead), zeros outside
            padded = np.zeros((len(x) + 2 * TAPS - 1, x.shape[1]), dtype=np.float32)
            padded[TAPS - 1:TAPS - 1 + len(x)] = x
            taps = np.concatenate((w[name + ".lookback_filter.weight"], w[name + ".lookahead_filter.weight"]), axis=1)
            view = sliding_window_view(padded, 2 * TAPS, axis=0)[:len(x)]
            return x + np.einsum("tpk,pk->tp", view, taps, optimize=True)

        x = (features - w["cmvn_mean"]) * w["cmvn_istd"]
        memory = fsmn(dense(dense(x, "dfsmn.fc1.0"), "dfsmn.fc2.0"), "dfsmn.fsmn1")
        for index in range(BLOCKS):
            name = f"dfsmn.fsmns.{index}"
            hidden = dense(dense(memory, name + ".fc1.0"), name + ".fc2", bias=False, relu=False)
            memory = fsmn(hidden, name + ".fsmn") + memory
        logits = dense(dense(memory, "dfsmn.dnns.0"), "out", relu=False)[:, 0]
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def probabilities(self, wav_path, workers, cancel, report):
        """Speech probability every 10 ms of a 16 kHz mono WAV file; pieces of the file run on several threads."""
        self.load()
        with wave.open(str(wav_path), "rb") as audio:
            total = audio.getnframes()
        count = 1 + (total - WINDOW) // SHIFT if total >= WINDOW else 0
        firsts = list(range(0, count, CHUNK_FRAMES))
        done, lock = [0], Lock()

        def piece(first):
            check_cancel(cancel)
            last = min(count, first + CHUNK_FRAMES)
            with wave.open(str(wav_path), "rb") as audio:
                audio.setpos(first * SHIFT)
                samples = np.frombuffer(audio.readframes((last - 1 - first) * SHIFT + WINDOW), dtype="<i2")
            result = self.forward(kaldi_fbank(samples))
            with lock:
                done[0] += last - first
                ratio = done[0] / count
            report(ratio)
            return result

        if not firsts:
            return np.zeros(0, dtype=np.float32)
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(firsts))), thread_name_prefix="firered-vad") as executor:
            return np.concatenate(list(executor.map(piece, firsts)))


def to_grid(probabilities, frames, frame_samples):
    """The 10 ms probabilities on a coarser grid of `frames` frames of `frame_samples` samples: the highest value
    of the 10 ms frames that start inside each."""
    grid = np.zeros(frames, dtype=np.float32)
    if len(probabilities) and frames:
        index = np.minimum(np.arange(len(probabilities)) * SHIFT // frame_samples, frames - 1)
        np.maximum.at(grid, index, probabilities)
    return grid
