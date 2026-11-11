"""Perception: merges the layers into one screen state and emits short text events.

Works like a video codec: a keyframe (window summary) when the context switches or when
a delta would cost more than the summary itself, otherwise only what changed.

Event line grammar (also explained to the model in brain.SYSTEM):
  window: proc «title»   keyframe, followed by the window's elements
  + / - / ~              appeared / disappeared / changed
  focus → ...            keyboard focus moved
  ▶ / ■                  motion (video, animation) started / stopped
  @proc                  change in a background window (toast, taskbar, ...)
"""
import ctypes
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from . import winapi
from .geom import Grid, Rect
from .uia import TEXT_ROLES, El, UiaReader, clip

# Window-frame buttons repeat in every app and say nothing about the content.
FRAME_BUTTONS = {"свернуть", "развернуть", "восстановить", "закрыть", "minimize", "maximize",
                 "restore", "close", "назад", "вперед", "back", "forward", "перезагрузить", "reload"}


@dataclass
class Event:
    t_frame: float   # perf_counter of the frame where the change was first seen
    t_emit: float
    kind: str        # window | ui | text | motion
    text: str

    @property
    def latency_ms(self) -> float:
        return (self.t_emit - self.t_frame) * 1000


@dataclass
class Win:
    hwnd: int
    process: str
    title: str
    rect: Rect
    els: list[El]
    t_snap: float
    targets: list[El] = field(default_factory=list)   # every element with its rect, for clicking


@dataclass
class View:
    """Read-only copy of what collect() needs, published after every step."""
    wins: dict
    fg: int
    text: list
    words: dict


@dataclass
class Motion:
    rect: Rect
    t_seen: float
    thumb: np.ndarray | None = None   # what the region looked like when the eyes last looked
    t_looked: float = 0.0


def thumb(crop: np.ndarray) -> np.ndarray:
    """Tiny 16x9 grayscale sketch of a region, enough to notice a scene change."""
    h, w = crop.shape[:2]
    return crop[:: max(h // 9, 1), :: max(w // 16, 1), 1][:9, :16].astype(np.float32)


def grows(old: str, new: str) -> bool:
    """True when one text extends the other: typing or deleting at the end."""
    return bool(old) and bool(new) and old != new and (new.startswith(old) or old.startswith(new))


def norm(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


class EchoFilter:
    """Ignores our own printed output when the terminal hosting us shows it back."""

    def __init__(self, host: int, ttl: float = 60.0):
        self.host, self.ttl = host, ttl
        self.lines: deque[tuple[float, str]] = deque(maxlen=300)

    def printed(self, text: str) -> None:
        t = time.monotonic()
        for ln in text.splitlines():
            n = norm(ln)
            if len(n) >= 6:
                self.lines.append((t, n))

    def is_echo(self, hwnd: int, s: str) -> bool:
        if hwnd != self.host or not self.lines:
            return False
        n = norm(s)
        if len(n) < 6:
            return False
        cutoff = time.monotonic() - self.ttl
        return any(t > cutoff and n in ln for t, ln in self.lines)


class Perception:
    def __init__(self, grid: Grid, uia: UiaReader, ocr=None, echo: EchoFilter | None = None,
                 snap_interval: float = 0.25, keyframe_lines: int = 60,
                 ticker_hits: int = 3, ticker_window: float = 10.0, motion_hold: float = 2.0,
                 relook_after: float = 2.0, scene_change: float = 25.0, monitors: list[dict] | None = None,
                 typing_interval: float = 1.0, exclude_titles: list[str] | None = None):
        self.grid, self.uia, self.ocr = grid, uia, ocr
        self.echo = echo or EchoFilter(0)
        self.snap_interval = snap_interval
        self.keyframe_lines = keyframe_lines
        self.ticker_hits, self.ticker_window = ticker_hits, ticker_window
        self.motion_hold = motion_hold
        self.relook_after, self.scene_change = relook_after, scene_change
        self.typing_interval = typing_interval
        self.sent: dict[tuple, str] = {}       # growing texts (typing): last value we reported
        self.t_sent: dict[tuple, float] = {}
        self.pending: dict[tuple, tuple[str, str, str, float]] = {}   # ident -> (label, text, prefix, t_first)
        self.looks: list[tuple[Rect, str, float]] = []   # regions the eyes should describe
        self.frame: np.ndarray | None = None
        self.monitors: list[tuple[Rect, str]] = []
        self._name_monitors(monitors or [])

        self.wins: dict[int, Win] = {}
        self.fg = 0
        self.dirty: dict[int, float] = {}            # hwnd -> frame time of oldest unread change
        self.text: list[tuple[int, Rect, str]] = []  # OCR lines (hwnd, rect, text)
        self.words: dict[tuple, tuple] = {}          # OCR line -> ((Rect, word), ...) for clicking
        self.t_step = 0.0                            # frame time of the last processed frame
        self.t_changed = 0.0                         # frame time of the last frame with changes
        self.t_cell = grid.empty().astype(np.float64)   # frame time of each cell's last change
        self.refresh_req: set[int] = set()           # windows whose targets the API wants re-read
        self.t_refreshed: dict[int, float] = {}
        self.view: View | None = None
        self.motion: list[Motion] = []
        self.hits: dict[tuple, deque] = {}
        self.tickers: set[tuple] = set()
        self.procs: dict[int, str] = {}
        self.own = {winapi.console_window()} - {0}
        self.exclude_titles = exclude_titles or []   # e.g. our own launcher window
        self.exclude_prefixes: list[str] = []        # set at runtime by an agent: «AI Vision…»
        self._refresh_own()
        self.t_prune = 0.0
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- main step
    def step(self, frame: np.ndarray, t_frame: float, changed: np.ndarray, hot: np.ndarray) -> list[Event]:
        events: list[Event] = []
        now = time.perf_counter()
        self.frame = frame
        self.t_step = t_frame
        if changed.any():
            self.t_changed = t_frame
            self.t_cell[changed] = t_frame
        # UI Automation lives in this thread (COM), so the API asks and we do the reading.
        for h in list(self.refresh_req):
            self.refresh_targets(h)
            self.t_refreshed[h] = now
        self.refresh_req.clear()

        fg = winapi.foreground()
        if fg and fg not in self.own and fg != self.fg:
            self.fg = fg
            w = self._snap(fg)
            fr = w.rect
            self.text = [x for x in self.text if not fr.contains(*x[1].center)]
            self._add(events, t_frame, "window", self._keyframe(w))
            self.dirty.pop(fg, None)

        owners = self._owners(changed)
        attrib = {h: self.grid.components(m) for h, m in owners.items()}
        for h in owners:
            self.dirty.setdefault(h, t_frame)

        for h, t0 in list(self.dirty.items()):
            old = self.wins.get(h)
            if old and now - old.t_snap < self.snap_interval:
                continue
            del self.dirty[h]
            if old is None:
                w = self._snap(h)
                if w.els or w.title:
                    self._add(events, t0, "window", self._keyframe(w, limit=12, tag=h != self.fg))
                continue
            self._add(events, t0, "ui", self._delta(old, self._snap(h), now))

        if self.ocr is not None and changed.any():
            covered = self._covered(attrib)
            stale = changed & covered
            if stale.any():
                self.text = [x for x in self.text if not self._in_mask(stale, x[1])]
            for h, m in owners.items():
                for band, zone in self._bands(h, m & ~covered):
                    self._add(events, t_frame, "text", self._read_text(frame, band, h, zone, now))

        self._add(events, t_frame, "motion", self._motion(frame, t_frame, hot, now))
        for t0, line in self._flush_typed(now):
            self._add(events, t0, "text", line)

        if now - self.t_prune > 2.0:
            self.t_prune = now
            self._refresh_own()
            self._add(events, t_frame, "window", self._prune())
        self.publish()
        return events

    def publish(self) -> None:
        """Freeze the click-target picture for readers that must not wait for a whole step.

        A step holds the lock through UI Automation and OCR (up to a few hundred ms); the
        cursor-routing API reads this copy instead, so a lookup stays in the tens of ms.
        """
        self.view = View(dict(self.wins), self.fg, list(self.text), dict(self.words))

    def _owners(self, mask: np.ndarray) -> dict[int, np.ndarray]:
        """Split changed cells by the top-level window that is visible at each cell."""
        out: dict[int, np.ndarray] = {}
        c = self.grid.cell
        for row, col in zip(*np.nonzero(mask)):
            h = winapi.root_window_at(*self.grid.cell_center(row, col))
            if not h or h in self.own:
                continue
            m = out.get(h)
            if m is None:
                m = out[h] = self.grid.empty()
            m[row, col] = True
        return out

    # ---------------------------------------------------------------- structure
    def _snap(self, h: int) -> Win:
        t = winapi.title(h)
        if h not in self.procs:
            self.procs[h] = winapi.process_name(h)
        els, targets = self.uia.read(h, t)
        w = Win(h, self.procs[h], t, winapi.rect(h), els, time.perf_counter(), targets)
        self.wins[h] = w
        return w

    def last_change(self, area: Rect | None = None) -> float:
        """Frame time of the last pixel change, on the whole screen or inside area.

        Waiting for the target window to calm down must not be blocked by a video or an
        animated chat on another monitor.
        """
        if area is None:
            return self.t_changed
        r = area.clip(self.grid.bounds)
        if r is None:
            return 0.0
        c, g = self.grid.cell, self.grid
        r0, c0 = (r.y - g.oy) // c, (r.x - g.ox) // c
        r1, c1 = -(-(r.y + r.h - g.oy) // c), -(-(r.x + r.w - g.ox) // c)
        cells = self.t_cell[r0:r1, c0:c1]
        return float(cells.max()) if cells.size else 0.0

    def dirty_in(self, area: Rect | None = None) -> bool:
        """Changed windows not re-read yet (only those overlapping area, if given)."""
        if area is None:
            return bool(self.dirty)
        for h in self.dirty:
            w = self.wins.get(h)
            if (w.rect if w else winapi.rect(h)).overlap(area):
                return True
        return False

    def refresh_targets(self, h: int) -> None:
        """Re-read click targets of a window without touching the event baseline.

        Chrome and Electron build their accessibility tree lazily, so a window that has not
        changed since its first, thin snapshot can miss most of its buttons.
        """
        w = self.wins.get(h)
        if w is not None:
            w.targets = self.uia.read(h, w.title)[1]
            self.publish()

    def _meaningful(self, w: Win) -> list[El]:
        out = []
        for e in w.els:
            if e.role == "btn" and norm(e.name) in FRAME_BUTTONS and e.rect.y - w.rect.y < 60:
                continue
            if self.echo.is_echo(w.hwnd, e.name):
                continue
            out.append(e)
        return out

    def _keyframe(self, w: Win, limit: int | None = None, tag: bool = False) -> str:
        limit = limit or self.keyframe_lines
        head = f"{'@' if tag else ''}window: {w.process} «{clip(w.title, 120)}»{self.where(w.rect)}"
        body = [e.render() + (" [focus]" if e.focused else "") for e in self._meaningful(w)]
        if len(body) > limit:
            body = body[:limit] + [f"… +{len(body) - limit} more"]
        return "\n".join([head] + body)

    def _delta(self, old: Win, new: Win, now: float) -> str:
        lines = []
        if new.title != old.title:
            lines.append(f"title «{clip(new.title, 120)}»")
        o_els, n_els = self._meaningful(old), self._meaningful(new)
        ok = {e.key: e for e in o_els}
        nk = {e.key: e for e in n_els}
        removed = [e for k, e in ok.items() if k not in nk]
        added = [e for k, e in nk.items() if k not in ok]

        changes = []
        for e in list(added):
            m = next((o for o in removed if o.role == e.role and
                      ((e.name and o.name == e.name) or self._same_place(o, e))), None)
            if m:
                removed.remove(m)
                added.remove(e)
                changes.append((m, e))

        prefix = f"@{new.process} " if new.hwnd != self.fg else ""
        for o, e in changes:
            if o.name == e.name and o.state == e.state and grows(o.value, e.value):
                label = f"{e.role} «{clip(e.name, 60)}»" if e.name else e.role
                line = self._typed(("val", new.hwnd, e.role, e.name), label, o.value, e.value, prefix, now)
                if line:
                    lines.append(line)
                continue
            ident = (new.hwnd, e.role, e.name) if o.name == e.name else (new.hwnd, e.role, *self._bucket(e.rect))
            state = self._tick(ident, now)
            if state == "muted":
                continue
            line = "~ " + self._change_text(o, e)
            lines.append(line + (" (keeps changing, muted)" if state == "new" else ""))
        # Elements that keep appearing and disappearing (hover, typing indicators) get muted.
        flaps = {e.key: self._tick(("flap", new.hwnd, e.key), now) for e in added + removed}
        added = [e for e in added if flaps[e.key] != "muted"]
        removed = [e for e in removed if flaps[e.key] != "muted"]
        mark = lambda e: " (flickers, muted)" if flaps[e.key] == "new" else ""
        lines += ["+ " + e.render() + mark(e) for e in added]
        for e in added:   # a big new picture has no text to read: ask the eyes
            if e.role == "img" and e.rect.w >= 120 and e.rect.h >= 120:
                self.looks.append((e.rect, f"img «{clip(e.name, 40)}»" if e.name else "img", now))
        if len(removed) > 5:
            lines.append(f"- {len(removed)} elements")
        else:
            lines += ["- " + e.render(60) + mark(e) for e in removed]

        nf = next((e for e in n_els if e.focused), None)
        if nf:
            line = self._settle(("focus", new.hwnd), "focus → " + nf.render(), prefix, now)
            if line:
                lines.append(line)

        if not lines:
            return ""
        tag = new.hwnd != self.fg
        key = self._keyframe(new, tag=tag)
        # Codec rule: when the delta is about as long as a fresh summary, send the summary.
        if len(lines) > 8 and sum(map(len, lines)) > 0.6 * len(key):
            return key
        return "\n".join(prefix + ln for ln in lines)

    @staticmethod
    def _change_text(o: El, e: El) -> str:
        if o.name == e.name and o.value != e.value:
            if o.value and e.value.startswith(o.value):
                return f"{e.role} «{clip(e.name, 60)}» += «{clip(e.value[len(o.value):], 120)}»"
            return e.render()
        if o.name != e.name and len(o.name) <= 40 and len(e.name) <= 40:
            return f"{e.role} «{o.name}» → «{e.name}»" + (f" = «{clip(e.value, 80)}»" if e.value else "")
        return e.render()

    @staticmethod
    def _bucket(r: Rect) -> tuple[int, int]:
        cx, cy = r.center
        return cx // 24, cy // 24

    def _same_place(self, a: El, b: El) -> bool:
        return self._bucket(a.rect) == self._bucket(b.rect)

    def _typed(self, ident: tuple, label: str, old: str, text: str, prefix: str, now: float) -> str | None:
        """Throttle a growing text (typing) to one update per interval; the latest value is
        flushed later, so nothing typed is lost and the model gets only what was added."""
        self.sent.setdefault(ident, old)
        if now - self.t_sent.get(ident, 0.0) >= self.typing_interval:
            return self._emit_typed(ident, label, text, now)
        t_first = self.pending.get(ident, ("", "", "", now))[3]
        self.pending[ident] = (label, text, prefix, t_first)
        return None

    def _settle(self, ident: tuple, line: str, prefix: str, now: float) -> str | None:
        """Report a state (e.g. focus) at most once per interval, and only if it differs
        from the last report, so flicker back and forth costs nothing."""
        if now - self.t_sent.get(ident, 0.0) >= self.typing_interval:
            self.pending.pop(ident, None)
            if self.sent.get(ident) == line:
                return None
            self.sent[ident], self.t_sent[ident] = line, now
            return line
        t_first = self.pending.get(ident, ("", "", "", now))[3]
        self.pending[ident] = ("", line, prefix, t_first)   # empty label marks a plain state
        return None

    def _emit_typed(self, ident: tuple, label: str, text: str, now: float) -> str:
        old = self.sent.get(ident, "")
        self.sent[ident], self.t_sent[ident] = text, now
        self.pending.pop(ident, None)
        if old and text.startswith(old):
            return f"~ {label} += «{clip(text[len(old):], 120)}»"
        return f"~ {label} = «{clip(text, 120)}»"

    def _flush_typed(self, now: float) -> list[tuple[float, str]]:
        out = []
        for ident, (label, text, prefix, t_first) in list(self.pending.items()):
            if now - self.t_sent.get(ident, 0.0) < self.typing_interval:
                continue
            if label:
                out.append((t_first, prefix + self._emit_typed(ident, label, text, now)))
                continue
            del self.pending[ident]
            if self.sent.get(ident) != text:
                self.sent[ident], self.t_sent[ident] = text, now
                out.append((t_first, prefix + text))
        return out

    def _tick(self, ident: tuple, now: float) -> str:
        """Mutes things that change all the time (timers, counters). Returns no | new | muted."""
        dq = self.hits.setdefault(ident, deque(maxlen=self.ticker_hits))
        if ident in self.tickers and dq and now - dq[-1] > self.ticker_window:
            self.tickers.discard(ident)
            dq.clear()
        dq.append(now)
        if ident in self.tickers:
            return "muted"
        if len(dq) == self.ticker_hits and now - dq[0] <= self.ticker_window:
            self.tickers.add(ident)
            return "new"
        return "no"

    def _refresh_own(self) -> None:
        """Windows we must not perceive: our console and windows showing our own output."""
        for t in self.exclude_titles:
            h = winapi.find_window(t)
            if h and h not in self.own:
                self._own(h)
        if self.exclude_prefixes:
            for h, w in list(self.wins.items()):
                if h not in self.own and any(w.title.startswith(x) for x in self.exclude_prefixes):
                    self._own(h)

    def _own(self, h: int) -> None:
        self.own.add(h)
        self.wins.pop(h, None)
        self.dirty.pop(h, None)
        self.text = [x for x in self.text if x[0] != h]

    def exclude(self, prefixes: list[str]) -> None:
        """Stop perceiving windows whose title starts with any prefix (an agent's own log)."""
        for x in prefixes:
            if len(x) >= 4 and x not in self.exclude_prefixes:
                self.exclude_prefixes.append(x)
        self._refresh_own()
        self.publish()

    def _prune(self) -> str:
        gone = [h for h in self.wins if not ctypes.windll.user32.IsWindow(h)]
        lines = []
        for h in gone:
            w = self.wins.pop(h)
            self.dirty.pop(h, None)
            self.text = [x for x in self.text if x[0] != h]
            if w.title:
                lines.append(f"closed: {w.process} «{clip(w.title, 80)}»")
        return "\n".join(lines)

    # ---------------------------------------------------------------- text (OCR)
    def _covered(self, attrib: dict[int, list[Rect]]) -> np.ndarray:
        """Cells whose content UI Automation already gave us as text."""
        s = 4
        g = self.grid
        canvas = np.zeros((g.rows * g.cell // s, g.cols * g.cell // s), bool)
        fgw = self.wins.get(self.fg)
        sources = [(fgw, [fgw.rect])] if fgw else []
        sources += [(self.wins[h], rs) for h, rs in attrib.items() if h != self.fg and h in self.wins]
        for w, clip_to in sources:
            for e in w.els:
                if e.role not in TEXT_ROLES or not e.rect.area:
                    continue
                if not any(e.rect.overlap(c) for c in clip_to):
                    continue
                x, y = e.rect.x - g.ox, e.rect.y - g.oy
                canvas[max(y // s, 0): max((y + e.rect.h) // s, 0),
                       max(x // s, 0): max((x + e.rect.w) // s, 0)] = True
        c = g.cell // s
        return canvas.reshape(g.rows, c, g.cols, c).mean(axis=(1, 3)) >= 0.25

    def _in_mask(self, mask: np.ndarray, r: Rect) -> bool:
        cell = self.grid.cell_at(*r.center)
        return cell is not None and bool(mask[cell])

    def _bands(self, h: int, mask: np.ndarray):
        """Turn one window's changed cells into full-width text bands, so OCR never sees half a line."""
        c = self.grid.cell
        wr = self.wins[h].rect if h in self.wins else winapi.rect(h)
        b = self.grid.bounds
        x0, x1 = max(wr.x, b.x), min(wr.x + wr.w, b.x + b.w)
        if x1 <= x0:
            return
        spans: list[list[int]] = []
        for r in sorted(self.grid.components(mask), key=lambda r: r.y):
            y0, y1 = r.y, r.y + r.h
            if spans and y0 <= spans[-1][1] + c:
                spans[-1][1] = max(spans[-1][1], y1)
            else:
                spans.append([y0, y1])
        for y0, y1 in spans:
            zone = Rect(x0, y0, x1 - x0, y1 - y0)
            top, bot = max(y0 - c, b.y), min(y1 + c, b.y + b.h)
            yield Rect(x0, top, x1 - x0, bot - top), zone

    def _read_text(self, frame: np.ndarray, band: Rect, h: int, zone: Rect, now: float) -> str:
        w = self.wins.get(h)
        known = set()
        for e in (w.els if w else []):
            known.update(norm(x) for x in f"{e.name} {e.value}".split())
        fresh = []
        crop, band = self.grid.crop(frame, band)
        if band is None:
            return ""
        for r, t, boxes in self.ocr.read(crop, band):
            n = norm(t)
            # Two letters are enough for a button («OK», «Да»); two of anything else is noise.
            short = len(n) < 3 and not (len(n) == 2 and n.isalpha())
            if short or not zone.contains(*r.center) or self.echo.is_echo(h, t):
                continue
            words = [norm(x) for x in t.split() if len(norm(x)) >= 2]
            if words and all(x in known for x in words):   # UI Automation already said this
                continue
            if winapi.root_window_at(*r.center) != h:   # band crossed into a window on top
                continue
            fresh.append((h, r, t))
            self.words[(h, r, t)] = boxes
        old = [x for x in self.text if x[0] == h and zone.contains(*x[1].center)]
        keep = [x for x in self.text if x not in old]
        self.text = keep + fresh
        if len(self.words) > 2 * len(self.text) + 500:
            live = set(self.text)
            self.words = {k: v for k, v in self.words.items() if k in live}

        old_t = {norm(x[2]): x for x in old}
        new_t = {norm(x[2]): x for x in fresh}
        gone = [x for k, x in old_t.items() if k not in new_t]
        came = [x for k, x in new_t.items() if k not in old_t]

        prefix = f"@{self.procs.get(h, '?')} " if h != self.fg else ""
        lines = []
        for x in came:
            twin = next((g for g in gone if abs(g[1].center[1] - x[1].center[1]) <= max(x[1].h, 8)
                         and g[1].overlap(x[1])), None)
            if twin:
                gone.remove(twin)
                if grows(twin[2], x[2]):
                    # Left edge stays put while a line is typed, so it identifies the line.
                    ident = ("ocr", h, x[1].x // 24, x[1].center[1] // 24)
                    line = self._typed(ident, "text", twin[2], x[2], prefix, now)
                    if line:
                        lines.append(line)
                    continue
                state = self._tick(("ocr", h, *self._bucket(x[1])), now)
                if state == "muted":
                    continue
                lines.append(f"~ «{clip(x[2], 120)}»" + (" (keeps changing, muted)" if state == "new" else ""))
            else:
                lines.append(f"+ «{clip(x[2], 120)}»")
        if len(gone) > 3:
            lines.append(f"- {len(gone)} text lines")
        else:
            lines += [f"- «{clip(x[2], 60)}»" for x in gone]
        if not lines:
            return ""
        return "\n".join(prefix + ln for ln in lines)

    # ---------------------------------------------------------------- motion
    def _motion(self, frame: np.ndarray, t_frame: float, hot: np.ndarray, now: float) -> str:
        min_area = 12 * self.grid.cell ** 2   # smaller than ~110x110 is a spinner, not content
        lines = []
        for r in self.grid.components(hot):
            if r.area < min_area:
                continue
            m = next((m for m in self.motion if m.rect.overlap(r)), None)
            if m:
                m.rect, m.t_seen = r, now
                # Look again only when the picture itself changed (a new scene), not every frame.
                sketch = thumb(self.grid.crop(frame, r)[0])
                if (now - m.t_looked >= self.relook_after and m.thumb is not None
                        and sketch.shape == m.thumb.shape
                        and float(np.abs(sketch - m.thumb).mean()) > self.scene_change):
                    m.thumb, m.t_looked = sketch, now
                    self.looks.append((r, "motion", t_frame))
            else:
                self.motion.append(Motion(r, now, thumb(self.grid.crop(frame, r)[0]), now))
                self.looks.append((r, "motion", t_frame))
                pct = 100 * r.area / (self.grid.width * self.grid.height)
                lines.append(f"▶ motion {r}{self.where(r)} ({pct:.0f}% of screen){self._what(r)}")
        for m in list(self.motion):
            if now - m.t_seen > self.motion_hold:
                self.motion.remove(m)
                lines.append(f"■ motion stopped {m.rect}")
        return "\n".join(lines)

    def _what(self, r: Rect) -> str:
        """Name the smallest known element under the middle of a region."""
        cx, cy = r.center
        w = self.wins.get(winapi.root_window_at(cx, cy))
        if not w:
            return ""
        inside = [e for e in w.els if e.name and e.rect.contains(cx, cy)]
        if not inside:
            return f" in {w.process}"
        e = min(inside, key=lambda e: e.rect.area)
        return f" in {e.role} «{clip(e.name, 60)}»"

    # ---------------------------------------------------------------- output
    def _add(self, events: list[Event], t: float, kind: str, text: str) -> None:
        if text:
            events.append(Event(t, time.perf_counter(), kind, text))

    def _name_monitors(self, mons: list[dict]) -> None:
        """Name monitors by position relative to the primary one; one monitor needs no name."""
        if len(mons) < 2:
            return
        prim = next((m for m in mons if m.get("is_primary")), mons[0])
        pr = Rect(prim["left"], prim["top"], prim["width"], prim["height"])
        for m in mons:
            r = Rect(m["left"], m["top"], m["width"], m["height"])
            if r == pr:
                name = "main"
            elif r.x + r.w <= pr.x:
                name = "left"
            elif r.x >= pr.x + pr.w:
                name = "right"
            else:
                name = "top" if r.y + r.h <= pr.y else "bottom"
            self.monitors.append((r, name))

    def where(self, r: Rect) -> str:
        """' [left monitor]' style tag for a screen rect; minimized windows sit off-screen."""
        if not self.monitors:
            return ""
        best = max(self.monitors, key=lambda m: m[0].overlap(r))
        return f" [{best[1]} monitor]" if best[0].overlap(r) else " [off-screen]"

    def foreground_rect(self) -> Rect | None:
        w = self.wins.get(self.fg)
        return w.rect.clip(self.grid.bounds) if w else None

    def now_line(self) -> str:
        """One-line anchor for the model: what is in front of the user right now."""
        w = self.wins.get(self.fg)
        if not w:
            return ""
        s = f"foreground {w.process} «{clip(w.title, 120)}»{self.where(w.rect)}"
        f = next((e for e in w.els if e.focused), None)
        if f:
            s += f"; focus {f.render(60)}"
        if self.motion:
            s += f"; motion in {len(self.motion)} region(s)"
        return s

    def render_state(self) -> str:
        """Full current picture, used as a fresh starting point for the model."""
        parts = []
        fgw = self.wins.get(self.fg)
        if fgw:
            parts.append("foreground " + self._keyframe(fgw))
        others = [w for h, w in self.wins.items() if h != self.fg and w.title
                  and ctypes.windll.user32.IsWindowVisible(h)]
        if others:
            parts.append("other windows: " + "; ".join(f"{w.process} «{clip(w.title, 60)}»" for w in others[:8]))
        if self.text:
            lines = sorted(self.text, key=lambda x: (x[1].y // 8, x[1].x))[:60]
            parts.append("screen text (OCR):\n" + "\n".join(clip(t, 120) for _, _, t in lines))
        if self.motion:
            parts.append("motion: " + "; ".join(f"{m.rect}{self._what(m.rect)}" for m in self.motion))
        return "\n".join(parts)
