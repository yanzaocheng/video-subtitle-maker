from pathlib import Path
import ctypes
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import winreg

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765"
SERVICE = "asr-web-siliconflow"
# Local health checks must not pass through a configured system proxy.
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def is_running():
    try:
        with HTTP.open(URL + "/api/health", timeout=2) as response:
            data = json.load(response)
            if not isinstance(data, dict) or data.get("service") != SERVICE:
                raise RuntimeError("8765 端口已被其他程序使用，请勿重复启动。")
            return True
    except urllib.error.HTTPError:
        raise RuntimeError("8765 端口已被其他程序使用，未检测到字幕服务。") from None
    except (ValueError, UnicodeError):
        raise RuntimeError("8765 端口的响应不是有效的字幕服务标识。") from None
    except (urllib.error.URLError, socket.timeout, ConnectionError):
        # An occupied but unresponsive port is not permission to launch another server.
        try:
            connection = socket.create_connection(("127.0.0.1", 8765), timeout=1)
        except OSError:
            return False
        connection.close()
        raise RuntimeError("8765 端口已占用，但字幕服务没有正常响应。请稍后再启动。") from None


def find_edge():
    candidates = []
    for variable in ("ProgramFiles(x86)", "ProgramFiles", "LOCALAPPDATA", "ProgramW6432"):
        directory = os.environ.get(variable)
        if directory:
            candidates.append(Path(directory) / "Microsoft" / "Edge" / "Application" / "msedge.exe")
    # Also support an existing installation registered at a nonstandard location.
    for hive in (winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE):
        for view in (winreg.KEY_WOW64_64KEY, winreg.KEY_WOW64_32KEY):
            try:
                with winreg.OpenKey(
                    hive, r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\msedge.exe",
                    0, winreg.KEY_READ | view,
                ) as key:
                    value, _ = winreg.QueryValueEx(key, None)
                    if isinstance(value, str):
                        candidates.append(Path(os.path.expandvars(value.strip().strip('"'))))
            except OSError:
                continue
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "未找到已安装的 Microsoft Edge。请检查 Edge 安装位置后重试；本程序不会自动安装浏览器。"
    )


def main():
    edge = find_edge()
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        raise RuntimeError("无法读取 LOCALAPPDATA，不能创建本机桌面配置目录。")
    app_data = Path(local_app_data) / "ASR-Web"
    profile = app_data / "desktop-profile"
    profile.mkdir(parents=True, exist_ok=True)
    if not is_running():
        python = Path(sys.executable).with_name("python.exe")
        if not python.is_file():
            raise RuntimeError(f"未找到后台 Python：{python}")
        logs = app_data / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        # No credentials or request bodies are included in the command or launcher log.
        with (logs / "server.log").open("ab", buffering=0) as output:
            server = subprocess.Popen(
                [str(python), "-s", "-u", str(ROOT / "app.py")],
                cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
        for _ in range(80):
            if is_running():
                break
            if server.poll() is not None:
                raise RuntimeError(f"本机服务启动后退出，请查看 {logs / 'server.log'}")
            time.sleep(0.25)
        else:
            raise RuntimeError(f"本机服务未能启动，请查看 {logs / 'server.log'}")
    subprocess.Popen(
        [str(edge), "--app=" + URL, "--user-data-dir=" + str(profile), "--no-first-run"],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        ctypes.windll.user32.MessageBoxW(None, str(exc), "视频字幕制作：启动失败", 0x10)
