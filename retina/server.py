"""Local HTTP API on 127.0.0.1.

GET  /state /events /stats
GET  /elements?scope=fg|all&area=x,y,w,h&hidden=1&exclude=title   click targets with screen rects
GET  /find?q=label&role=btn&limit=5&scope=&area=&exclude=title      best targets for a label, no model
GET  /events?since=seq                               events after seq: [{seq, kind, text}], last
POST /exclude {"prefixes": ["AI Vision"]}           stop perceiving windows whose title starts so
GET  /settled?quiet=300&timeout=5000&expect=1&scope=fg|area=x,y,w,h
                                                     block until the screen (or that area) stops changing
POST /ask {"q": ...}   POST /look {"rect": [x, y, w, h]}
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

from . import index, winapi
from .geom import Rect


def _area(qs: dict) -> Rect | None:
    v = qs.get("area", [""])[0]
    if not v:
        return None
    x, y, w, h = (int(float(p)) for p in v.split(","))
    return Rect(x, y, w, h)


def _num(qs: dict, key: str, default: float) -> float:
    try:
        return float(qs.get(key, [default])[0])
    except (TypeError, ValueError):
        return default


def serve(per, brain, eyes, echo, port: int, recent, stats, journal=None) -> None:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            url = urlsplit(self.path)
            qs = parse_qs(url.query)
            if url.path == "/elements":
                self._elements(qs)
                return
            if url.path == "/find":
                self._find(qs)
                return
            if url.path == "/settled":
                self._settled(qs)
                return
            if self.path == "/state":
                with per.lock:
                    state = per.render_state()
                self._send(200, {"state": state})
            elif url.path == "/events":
                since = qs.get("since", [None])[0]
                if since is None or journal is None:
                    self._send(200, {"events": list(recent)})
                else:
                    items = list(journal)
                    n = int(_num(qs, "since", 0))
                    self._send(200, {"events": [{"seq": s, "kind": k, "text": t} for s, k, t in items if s > n],
                                     "last": items[-1][0] if items else 0})
            elif self.path == "/stats":
                self._send(200, stats())
            else:
                self._send(404, {"error": "unknown path"})

        def do_POST(self) -> None:
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n).decode("utf-8") or "{}") if n else {}
            if self.path == "/look":
                self._look(body)
                return
            if self.path == "/exclude":
                with per.lock:
                    per.exclude([str(x) for x in body.get("prefixes", [])])
                    self._send(200, {"prefixes": per.exclude_prefixes})
                return
            if self.path != "/ask":
                self._send(404, {"error": "unknown path"})
                return
            if brain is None:
                self._send(400, {"error": "start watch with --model to enable the brain"})
                return
            r = brain.ask(body.get("q", ""))
            echo.printed(r["answer"])   # the answer may show up on screen; don't perceive it as news
            self._send(200, r)

        def _targets(self, qs: dict) -> list:
            scope, area, exclude = qs.get("scope", ["all"])[0], _area(qs), qs.get("exclude", [])
            view = per.view                  # published after each step: no waiting on the lock
            if view is None:
                with per.lock:
                    return index.collect(per, scope, area, winapi.root_window_at, exclude)
            return index.collect(view, scope, area, winapi.root_window_at, exclude)

        def _fg(self) -> dict:
            view = per.view
            if view is None:
                with per.lock:
                    w = per.wins.get(per.fg)
            else:
                w = view.wins.get(view.fg)
            if not w:
                return {}
            return {"process": w.process, "title": w.title, "rect": [w.rect.x, w.rect.y, w.rect.w, w.rect.h]}

        def _refresh_fg(self, timeout: float = 1.5) -> bool:
            """Ask the watch loop to re-read the foreground window's targets and wait for it."""
            t0 = time.perf_counter()
            with per.lock:
                h = per.fg
                if h not in per.wins:
                    return False
                per.refresh_req.add(h)
            while time.perf_counter() - t0 < timeout:
                time.sleep(0.02)
                with per.lock:
                    if per.t_refreshed.get(h, 0) >= t0:
                        return True
            return False

        def _elements(self, qs: dict) -> None:
            hidden = qs.get("hidden", ["0"])[0] == "1"
            if qs.get("refresh", ["0"])[0] == "1":
                self._refresh_fg()
            ts = [t.to_dict() for t in self._targets(qs) if hidden or t.visible]
            self._send(200, {"elements": ts, "foreground": self._fg()})

        def _find(self, qs: dict) -> None:
            q = qs.get("q", [""])[0]
            if not q:
                self._send(400, {"error": "q is required"})
                return
            role = qs.get("role", [None])[0]
            limit = int(_num(qs, "limit", 5))
            hidden = qs.get("hidden", ["0"])[0] == "1"
            r = index.find(self._targets(qs), q, role, limit, hidden)
            if not r["matches"]:
                # The foreground window may still hold its first, thin accessibility tree.
                if self._refresh_fg():
                    r = index.find(self._targets(qs), q, role, limit, hidden)
                    r["refreshed"] = True
            r["foreground"] = self._fg()
            self._send(200, r)

        def _settled(self, qs: dict) -> None:
            quiet = _num(qs, "quiet", 300) / 1000
            timeout = _num(qs, "timeout", 5000) / 1000
            expect = qs.get("expect", ["0"])[0] == "1"
            area = _area(qs)
            if area is None and qs.get("scope", [""])[0] == "fg":
                with per.lock:
                    w = per.wins.get(per.fg)
                    area = w.rect if w else None
            t_req = time.perf_counter()
            while True:
                with per.lock:
                    t_step, t_changed, dirty = per.t_step, per.last_change(area), per.dirty_in(area)
                    motion = len(per.motion) if area is None else sum(1 for m in per.motion if m.rect.overlap(area))
                now = time.perf_counter()
                st = index.settle_state(t_req, now, t_step, t_changed, dirty, quiet, expect)
                if st == "settled" or now - t_req >= timeout:
                    self._send(200, {"settled": st == "settled", "waited_ms": round((now - t_req) * 1000),
                                     "changed": t_changed > t_req, "motion": motion,
                                     "quiet_ms": round((now - t_changed) * 1000)})
                    return
                time.sleep(0.02)

        def _look(self, body: dict) -> None:
            """Describe a region now (default: the foreground window) and log it as an event."""
            if eyes is None:
                self._send(400, {"error": "start watch with --eyes to enable vision"})
                return
            from .geom import Rect
            from .vlm import rgb
            with per.lock:
                frame = per.frame
                r = Rect(*body["rect"]) if body.get("rect") else per.foreground_rect()
                crop, r = per.grid.crop(frame, r) if frame is not None and r else (None, None)
            if r is None:
                self._send(409, {"error": "nothing to look at yet"})
                return
            t = time.perf_counter()
            text, tokens = eyes.describe(rgb(crop))
            eyes.out.put((t, f"seen{per.where(r)} {r}: {text}"))
            self._send(200, {"text": text, "rect": str(r), "image_tokens": tokens,
                             "ms": (time.perf_counter() - t) * 1000})

        def log_message(self, *args) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
