"""Desktop launcher: pick a model, start/stop the watcher, watch live stats and events, ask.

The watcher runs as a child process with --quiet; this window talks to its local API.
The launcher window is excluded from perception, so the model never reads its own feed.
"""
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent.parent
TITLE = "screen-retina"
PORT = 8765
PREFERRED = ("qwen3-vl:4b-instruct", "qwen2.5:7b")


def _api(path: str, payload: dict | None = None, timeout: float = 2.0):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


class Api:
    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def models(self) -> dict:
        try:
            out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10,
                                 creationflags=subprocess.CREATE_NO_WINDOW).stdout
        except (OSError, subprocess.TimeoutExpired):
            return {"models": [], "default": ""}
        names = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip()]
        default = next((m for m in PREFERRED if m in names), names[0] if names else "")
        return {"models": names, "default": default}

    def start(self, model: str, eyes: bool) -> dict:
        if self.proc and self.proc.poll() is None:
            return {"ok": True}
        cmd = [sys.executable, "-m", "retina", "watch", "--quiet", "--port", str(PORT),
               "--exclude-title", TITLE]
        if model:
            cmd += ["--model", model]
        if model and eyes:
            cmd += ["--eyes", "same"]
        (ROOT / "data").mkdir(exist_ok=True)
        log = open(ROOT / "data" / "watch.log", "w", encoding="utf-8")
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                     creationflags=subprocess.CREATE_NO_WINDOW)
        return {"ok": True}

    def stop(self) -> dict:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None
        return {"ok": True}

    def poll(self) -> dict:
        """One call per second from the page: process state, stats and recent events."""
        alive = bool(self.proc and self.proc.poll() is None)
        try:
            stats = _api("/stats")
            events = _api("/events")["events"]
        except OSError:
            stats, events = None, []
        error = ""
        if self.proc and not alive:
            error = self._log_tail()
        return {"running": alive, "ready": stats is not None, "stats": stats,
                "events": events[-150:], "error": error}

    def ask(self, question: str) -> dict:
        try:
            return _api("/ask", {"q": question}, timeout=120)
        except Exception as e:
            return {"error": str(e)}

    def look(self) -> dict:
        try:
            return _api("/look", {}, timeout=180)
        except Exception as e:
            return {"error": str(e)}

    def _log_tail(self) -> str:
        try:
            lines = (ROOT / "data" / "watch.log").read_text(encoding="utf-8").splitlines()
            return "\n".join(lines[-6:])
        except OSError:
            return "watcher exited"


def main() -> None:
    api = Api()
    window = webview.create_window(TITLE, str(Path(__file__).parent / "ui" / "index.html"),
                                   js_api=api, width=1180, height=780, min_size=(940, 620),
                                   background_color="#f6f7fb")
    window.events.closed += api.stop
    webview.start()


if __name__ == "__main__":
    main()
