"""One-time setup for local GPU speech recognition (isolated from the main Python environment)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
ENV = ROOT / "gpu-env"
MODELS = ROOT / "models"
DOWNLOADS = ROOT / "gpu-setup"
# The GPU environment is created from the Python that runs this script (the PyTorch wheel below needs 3.13).
BASE_PYTHON = Path(sys.executable)
UV = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv.exe")
CHUNK = 16 * 1024 * 1024
WORKERS = 12

TORCH = {
    "name": "torch-2.11.0+cu130-cp313-cp313-win_amd64.whl",
    "url": "https://download-r2.pytorch.org/whl/cu130/torch-2.11.0%2Bcu130-cp313-cp313-win_amd64.whl",
    "size": 1915220980,
    "sha256": "339801f2163698a53c7fb3c91883e7f44331d22c34d45acfbce4eff71f2332fa",
}
SMALL = [("chat_template.jinja", 1434, "491fff331513a97316daf53591c7b9b9da53ef8c"),
         ("generation_config.json", 165, "3ed98c80087572bad932a83279c5602e58bbfc5c"),
         ("processor_config.json", 487, "a07dff48032bed341260501d6688036c6fd7f3ef"),
         ("tokenizer_config.json", 998, "aa62b6c06ab8a704dc0895420fb1d2515954a49c")]
TOKENIZER = ("tokenizer.json", 11429653, "fe1fad59be22a41ee293363fcf95fdedbc7c93f3b49270b1d2e18bd1399a7a05")
REPOS = {
    "Qwen3-ASR-0.6B-hf": ("Qwen/Qwen3-ASR-0.6B-hf", "7f1569a48a89f3e3f4dc3a5c9d28bddd903bc76c",
                          ("config.json", 2398, "3b4221f651b59ec21f545e5852b4d6cb23420b09"),
                          ("model.safetensors", 1564928088, "d3f212dd20abecd315d830bc54ae3865e56ebfc3276484e57b771288ba27fd35")),
    "Qwen3-ASR-1.7B-hf": ("Qwen/Qwen3-ASR-1.7B-hf", "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
                          ("config.json", 2399, "331292992b1933c69fc83d77f0de560ab122a0d5"),
                          ("model.safetensors", 4076193080, "2db53c7d81bd9b8cbc6a074e89be2c968a0d373fb4ee68bb1b1e14f7042dfee1")),
}


def log(**values):
    print(json.dumps(values, ensure_ascii=False), flush=True)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def fetch_range(url, part, start, end):
    expected = end - start + 1
    for attempt in range(1, 15):
        have = part.stat().st_size if part.exists() else 0
        if have == expected:
            return
        if have > expected:
            part.unlink()
            have = 0
        request = urllib.request.Request(url, headers={"Range": f"bytes={start + have}-{end}", "Accept-Encoding": "identity"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, part.open("ab") as output:
                if response.status != 206:
                    raise ConnectionError(f"range not honoured: HTTP {response.status}")
                while block := response.read(1024 * 1024):
                    output.write(block)
            if part.stat().st_size == expected:
                return
        except Exception as exc:
            time.sleep(min(2 * attempt, 20))
            if attempt == 14:
                raise RuntimeError(f"download failed for {part.name}: {exc}") from exc
    raise RuntimeError(f"incomplete part {part.name}")


def download(url, target, size, digest, verify):
    if target.exists() and target.stat().st_size == size and verify(target) == digest:
        log(step="verified", file=target.name)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if size < CHUNK:
        with urllib.request.urlopen(url, timeout=120) as response:
            data = response.read()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
    else:
        parts = target.parent / (target.name + ".parts")
        parts.mkdir(exist_ok=True)
        count = (size + CHUNK - 1) // CHUNK
        started, done = time.monotonic(), 0
        with ThreadPoolExecutor(WORKERS) as pool:
            futures = [pool.submit(fetch_range, url, parts / f"{i:05d}", i * CHUNK, min((i + 1) * CHUNK, size) - 1) for i in range(count)]
            for future in as_completed(futures):
                future.result()
                done += 1
                received = sum(p.stat().st_size for p in parts.iterdir())
                if done % 8 == 0 or done == count:
                    log(step="downloading", file=target.name, percent=round(received / size * 100, 1),
                        mib_per_s=round(received / 2**20 / max(time.monotonic() - started, 0.1), 2))
        temporary = target.with_suffix(target.suffix + ".tmp")
        with temporary.open("wb") as output:
            for i in range(count):
                with (parts / f"{i:05d}").open("rb") as part:
                    shutil.copyfileobj(part, output, 8 * 1024 * 1024)
    if temporary.stat().st_size != size or verify(temporary) != digest:
        temporary.unlink()
        raise RuntimeError(f"integrity check failed: {target.name}")
    os.replace(temporary, target)
    if size >= CHUNK:
        shutil.rmtree(target.parent / (target.name + ".parts"))
    log(step="verified", file=target.name)


def download_model(folder):
    repo, revision, config, weights = REPOS[folder]
    destination = MODELS / folder
    base = f"https://huggingface.co/{repo}/resolve/{revision}/"
    for name, size, blob in [*SMALL, config]:
        download(base + name, destination / name, size, blob, git_blob)
    for name, size, digest in (TOKENIZER, weights):
        download(base + name, destination / name, size, digest, sha256)
    (destination / "verified.json").write_text(json.dumps({"repo": repo, "revision": revision}), encoding="utf-8")


def run(command):
    env = os.environ.copy()
    env.update({"UV_HTTP_TIMEOUT": "300", "PYTHONNOUSERSITE": "1"})
    subprocess.run(command, check=True, env=env)


def main():
    targets = sys.argv[1:] or ["env", "Qwen3-ASR-0.6B-hf", "Qwen3-ASR-1.7B-hf"]
    if "env" in targets:
        if sys.version_info[:2] != (3, 13):
            raise SystemExit("请用 Python 3.13 运行本脚本：PyTorch 安装包是 cp313 版本。")
        if not Path(UV).is_file():
            raise SystemExit("未找到 uv，请先安装：https://docs.astral.sh/uv/")
        wheel = DOWNLOADS / TORCH["name"]
        download(TORCH["url"], wheel, TORCH["size"], TORCH["sha256"], sha256)
        if not (ENV / "Scripts" / "python.exe").exists():
            run([UV, "venv", "--python", str(BASE_PYTHON), str(ENV)])
        python = str(ENV / "Scripts" / "python.exe")
        run([UV, "pip", "install", "--python", python, str(wheel)])
        run([UV, "pip", "install", "--python", python, "transformers>=5.13,<6", "accelerate", "numpy", "safetensors"])
        log(step="environment_ready", python=python)
    for folder in targets:
        if folder in REPOS:
            download_model(folder)
            log(step="model_ready", model=folder)
    log(step="done")


if __name__ == "__main__":
    main()
