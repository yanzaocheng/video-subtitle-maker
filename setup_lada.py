"""One-time setup for mosaic removal: downloads Lada (a separate AGPL-3.0 program) and unpacks it into lada/.

The app only starts lada-cli.exe as an external process; no Lada code is copied into or imported by this project.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess

from setup_local_gpu import ROOT, download, log, sha256

VERSION = "v0.11.0"
BASE_URL = f"https://github.com/ladaapp/lada/releases/download/{VERSION}/"
# Checksums from the release announcement (also the GitHub asset digests).
PARTS = [("lada-v0.11.0_windows_nvidia.7z.001", 2096103424, "861caf4bc3fb08bb4f145a0ef53172d051d39401e9f5b1c6cbab7206b32e518b"),
         ("lada-v0.11.0_windows_nvidia.7z.002", 401197306, "472b8012f676cca0ef0eb6af9a69ba1256370dbe6c1c84740ab34d4c2650b796")]
DOWNLOADS = ROOT / "lada-setup"
TARGET = ROOT / "lada"
STAGING = ROOT / "lada.tmp"


def seven_zip():
    candidates = [shutil.which("7z")]
    for variable in ("ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable]) / "7-Zip" / "7z.exe"))
    return next((c for c in candidates if c and Path(c).is_file()), None)


def main():
    if (TARGET / "verified.json").is_file() and (TARGET / "lada-cli.exe").is_file():
        log(step="already_installed", cli=str(TARGET / "lada-cli.exe"))
        return
    tool = seven_zip()
    if tool is None:
        raise SystemExit("未找到 7-Zip，请先安装：https://7-zip.org/")
    for name, size, digest in PARTS:
        download(BASE_URL + name, DOWNLOADS / name, size, digest, sha256)
    shutil.rmtree(STAGING, ignore_errors=True)
    log(step="extracting")
    subprocess.run([tool, "x", str(DOWNLOADS / PARTS[0][0]), f"-o{STAGING}", "-y", "-bso0", "-bsp0"], check=True)
    cli = next(STAGING.rglob("lada-cli.exe"), None)
    if cli is None:
        raise SystemExit("压缩包里没有 lada-cli.exe，可能下载的版本不对。")
    shutil.rmtree(TARGET, ignore_errors=True)
    os.replace(cli.parent, TARGET)
    shutil.rmtree(STAGING, ignore_errors=True)
    (TARGET / "verified.json").write_text(json.dumps({"version": VERSION, "parts": [p[0] for p in PARTS]}), encoding="utf-8")
    # The archives (about 2.5 GB) are not needed once unpacked.
    shutil.rmtree(DOWNLOADS, ignore_errors=True)
    log(step="done", cli=str(TARGET / "lada-cli.exe"))


if __name__ == "__main__":
    main()
