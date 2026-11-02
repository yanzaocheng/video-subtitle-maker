"""Local GPU speech recognition worker.

Runs inside the isolated gpu-env and exchanges one JSON object per line over stdin/stdout,
so the main application never imports PyTorch or holds GPU memory itself.
"""
import base64
import io
import json
import os
import sys
import time
import wave

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np


def emit(**values):
    sys.stdout.write(json.dumps(values, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def decode(data):
    with wave.open(io.BytesIO(base64.b64decode(data)), "rb") as audio:
        if (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) != (1, 2, 16000):
            raise ValueError("音频必须是 16kHz 单声道 16 位 PCM")
        pcm = audio.readframes(audio.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def main():
    model_dir = sys.argv[1]
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    if not torch.cuda.is_available():
        emit(fatal="未检测到可用的 NVIDIA 显卡（CUDA），无法使用本地 GPU 识别。")
        return 2
    free_before, total = torch.cuda.mem_get_info()
    started = time.monotonic()
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(model_dir, dtype=torch.bfloat16, local_files_only=True)
    model = model.to("cuda").eval()
    emit(ready=True, gpu=torch.cuda.get_device_name(0), total_mb=total // 2**20, free_before_mb=free_before // 2**20,
         free_after_mb=torch.cuda.mem_get_info()[0] // 2**20, load_seconds=round(time.monotonic() - started, 1))
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        ident = request.get("id")
        try:
            audios = [decode(item) for item in request["audios"]]
            language = request.get("language")
            inputs = processor.apply_transcription_request(
                audio=audios, language=[language] * len(audios) if language else None,
            ).to("cuda", torch.bfloat16)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=320, do_sample=False)
            texts = processor.decode(output[:, inputs["input_ids"].shape[1]:], return_format="transcription_only")
            emit(id=ident, texts=[text.strip() for text in texts])
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            emit(id=ident, oom=True)
        except Exception as exc:
            emit(id=ident, error=f"{type(exc).__name__}: {exc}"[:600])
    return 0


if __name__ == "__main__":
    sys.exit(main())
