"""CLI: python -m retina watch | ask "question" | state"""
import argparse
import json
import sys
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

# 1920x1080 at ~1 token per 28x28 patch: what one screenshot costs a vision model.
SCREENSHOT_TOKENS = 1920 * 1080 // 784


def est_tokens(text: str) -> int:
    return max(1, len(text) // 3)


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(p * len(s)))]


def watch(args) -> None:
    from . import winapi
    winapi.set_dpi_aware()
    from .capture import PixelDiff
    from .perception import EchoFilter, Event, Perception
    from .uia import UiaReader

    px = PixelDiff(cell=args.cell)
    ocr = None
    if not args.no_ocr:
        from .ocr import Ocr
        ocr = Ocr(args.lang)
    echo = EchoFilter(winapi.foreground())
    per = Perception(px.grid, UiaReader(), ocr, echo, monitors=px.sct.monitors[1:],
                     exclude_titles=args.exclude_title)

    def state() -> str:
        with per.lock:
            return per.render_state()

    def now() -> str:
        with per.lock:
            return per.now_line()

    brain = None
    if args.model:
        from .brain import Brain
        brain = Brain(args.model, state, now, num_ctx=args.ctx)

    period = 1.0 / args.fps
    shot_tokens = px.grid.width * px.grid.height // 784   # screenshots of every monitor, once a second
    t_start = time.perf_counter()
    st = {"ticks": 0, "events": 0, "event_tokens": 0, "seq": 0}
    busy: deque[float] = deque(maxlen=300)
    lat: deque[float] = deque(maxlen=2000)
    recent: deque[str] = deque(maxlen=200)
    journal: deque[tuple[int, str, str]] = deque(maxlen=1000)   # (seq, kind, text) for /events?since=
    vlat: deque[float] = deque(maxlen=200)

    def stats() -> dict:
        el = time.perf_counter() - t_start
        base = int(el) * shot_tokens
        d = {**st, "seconds": round(el), "screenshot_tokens_1fps": base,
             "times_fewer": round(base / max(st["event_tokens"], 1), 1),
             "tick_ms_avg": round(sum(busy) / max(len(busy), 1)), "tick_ms_max": round(max(busy, default=0)),
             "latency_ms_p50": round(pct(list(lat), .5)), "latency_ms_p95": round(pct(list(lat), .95)),
             "frame_wait_ms_max": round(period * 1000)}
        if brain:
            d.update(brain_new_tokens=brain.new_tokens, brain_cached_len=brain.cached_len,
                     brain_compactions=brain.compactions,
                     brain_error=brain.error)
        if eyes:
            d.update(eyes_looks=eyes.looks, eyes_image_tokens=eyes.image_tokens, eyes_error=eyes.error,
                     vision_latency_ms_p50=round(pct(list(vlat), .5)))
        return d

    eyes = None
    if args.eyes:
        from .vlm import Eyes
        eyes = Eyes(args.model if args.eyes == "same" else args.eyes, num_ctx=args.ctx)

    if args.port:
        from .server import serve
        serve(per, brain, eyes, echo, args.port, recent, stats, journal)

    Path("data").mkdir(exist_ok=True)
    log = open(Path("data") / f"events-{datetime.now():%Y%m%d-%H%M%S}.jsonl", "a", encoding="utf-8")
    t_stats = t_start
    api = f", api=http://127.0.0.1:{args.port}" if args.port else ""
    print(f"retina: watching {px.grid.width}x{px.grid.height} at {args.fps} fps, "
          f"OCR={ocr.lang if ocr else 'off'}, brain={args.model or 'off'}, eyes={args.eyes or 'off'}{api}", flush=True)

    try:
        while True:
            t0 = time.perf_counter()
            frame, t_frame = px.grab()
            changed, hot = px.diff(frame)
            with per.lock:
                events = per.step(frame, t_frame, changed, hot)
                looks, per.looks = per.looks, []
            if eyes:
                for r, label, t in looks:
                    crop, r = px.grid.crop(frame, r)
                    if r is not None:
                        eyes.look(crop, r, label + per.where(r), t)
                while not eyes.out.empty():
                    t_seen, text = eyes.out.get_nowait()
                    events.append(Event(t_seen, time.perf_counter(), "vision", text))
                    vlat.append(events[-1].latency_ms)
            for e in events:
                stamp = datetime.now().strftime("%H:%M:%S")
                line = f"[{stamp}] {e.text}"
                st["events"] += 1
                st["event_tokens"] += est_tokens(line)
                # Vision has its own clock (vlat); the first tick is the one-off initial look.
                if e.kind != "vision" and st["ticks"] > 0:
                    lat.append(e.latency_ms)
                recent.append(line)
                st["seq"] += 1
                journal.append((st["seq"], e.kind, e.text))
                log.write(json.dumps({"t": stamp, "kind": e.kind, "ms": round(e.latency_ms), "text": e.text},
                                     ensure_ascii=False) + "\n")
                if brain:
                    brain.ingest(line)
                if not args.quiet:
                    out = f"{line}   ({e.latency_ms:.0f} ms)"
                    echo.printed(out)
                    print(out, flush=True)
            log.flush()
            st["ticks"] += 1
            busy.append((time.perf_counter() - t0) * 1000)

            if time.perf_counter() - t_stats >= args.stats:
                t_stats = time.perf_counter()
                d = stats()
                b = (f" | brain evaluated {d['brain_new_tokens']:,} new tok (log in cache {d['brain_cached_len']:,})"
                     if brain else "")
                msg = (f"[stats {d['seconds']}s] ticks {d['ticks']} (avg {d['tick_ms_avg']} ms, max {d['tick_ms_max']}) | "
                       f"events {d['events']} ≈{d['event_tokens']:,} tok vs screenshots@1fps "
                       f"{d['screenshot_tokens_1fps']:,} tok (×{d['times_fewer']:.0f} less) | latency p50 "
                       f"{d['latency_ms_p50']} ms, p95 {d['latency_ms_p95']} ms, +≤{d['frame_wait_ms_max']} ms frame wait{b}")
                echo.printed(msg)
                print(msg, flush=True)
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        log.close()


def call(port: int, path: str, payload: dict | None = None):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}",
                                 data=json.dumps(payload).encode() if payload is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="retina")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("watch", help="watch the screen and emit events")
    w.add_argument("--fps", type=float, default=10)
    w.add_argument("--cell", type=int, default=32)
    w.add_argument("--lang", default=None, help="OCR language tag, default: user profile")
    w.add_argument("--no-ocr", action="store_true")
    w.add_argument("--model", default=None, help="Ollama model for the brain, e.g. qwen2.5:3b")
    w.add_argument("--eyes", default=None, help="Ollama vision model for pictures/video, or 'same' as --model")
    w.add_argument("--ctx", type=int, default=8192)
    w.add_argument("--port", type=int, default=8765)
    w.add_argument("--quiet", action="store_true", help="do not print events")
    w.add_argument("--stats", type=float, default=15, help="seconds between stats lines")
    w.add_argument("--exclude-title", action="append", default=[],
                   help="ignore windows with this exact title (e.g. a UI showing our events)")
    sub.add_parser("ui", help="open the desktop launcher")
    a = sub.add_parser("ask", help="ask the brain about the screen")
    a.add_argument("question")
    a.add_argument("--port", type=int, default=8765)
    lk = sub.add_parser("look", help="describe the foreground window (or --rect x y w h) with the eyes")
    lk.add_argument("--rect", type=int, nargs=4, default=None)
    lk.add_argument("--port", type=int, default=8765)
    s = sub.add_parser("state", help="print the current screen state")
    s.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    if args.cmd == "watch":
        watch(args)
    elif args.cmd == "ui":
        from .launcher import main as launcher_main
        launcher_main()
    elif args.cmd == "ask":
        r = call(args.port, "/ask", {"q": args.question})
        print(r["answer"])
        print(f"\n(first token {r['ttft_ms']:.0f} ms, full answer {r['ms']:.0f} ms | evaluated {r['prompt_tokens']} new tok "
              f"of {r['prompt_len']} in prompt | answer {r['answer_tokens']} tok)")
    elif args.cmd == "look":
        r = call(args.port, "/look", {"rect": args.rect})
        print(r["text"])
        print(f"\n({r['rect']}, {r['ms']:.0f} ms, image {r['image_tokens']} tok "
              f"vs full screenshot {SCREENSHOT_TOKENS} tok)")
    elif args.cmd == "state":
        r = call(args.port, "/state")
        print(r["state"])
        print(f"\n(≈{est_tokens(r['state'])} tok vs one screenshot {SCREENSHOT_TOKENS} tok)")


if __name__ == "__main__":
    main()
