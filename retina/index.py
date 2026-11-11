"""Click targets: where on screen a label is, so a cursor can be sent there.

Perception keeps a live picture for the event log; this module flattens it into targets
with screen rects (UI Automation elements of every window, OCR lines with word boxes)
and finds the best match for a label. No model is involved: matching is text folding
plus fuzzy similarity, so a lookup costs microseconds instead of a vision call.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher

from .geom import Rect

# Roles a person clicks; a plain text label of the same words is a worse target.
INTERACTIVE = {"btn", "splitbtn", "link", "menuitem", "item", "treeitem", "tab", "check",
               "radio", "combo", "edit", "headeritem", "cell", "spinner", "slider"}

MIN_SCORE = 0.6
MAX_WORDS = 8          # longest word run of an OCR line tried as a label
GAP_K = 1.6            # a gap wider than this many line heights splits glued menu items
PROSE_WORDS = 5        # a segment longer than this is a sentence, not a control label
PROSE_PENALTY = 0.3


def fold(s: str) -> str:
    """Lowercase, ё→е, punctuation to spaces: «Войти!» and «войти» are the same label."""
    s = s.lower().replace("ё", "е")
    return " ".join("".join(ch if ch.isalnum() else " " for ch in s).split())


def similarity(query: str, text: str) -> float:
    """How well a label on screen answers a query, 0..1."""
    q, t = fold(query), fold(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0
    pq, pt = f" {q} ", f" {t} "
    if pt.startswith(pq) or pt.endswith(pq):                # «Отправить» in «Отправить сейчас»
        return 0.85 + 0.1 * len(q) / len(t)
    if pq in pt:
        return 0.75 + 0.15 * len(q) / len(t)
    if len(t) >= 3 and pt in pq:                            # «кнопка Войти» → «Войти»
        return 0.7 + 0.15 * len(t) / len(q)
    if min(len(q), len(t)) >= 4:                            # OCR typos: «Отправигь»
        r = SequenceMatcher(None, q, t).ratio()
        if r >= 0.8:
            return 0.85 * r
    return 0.0


@dataclass
class Target:
    role: str
    name: str
    value: str
    state: str
    rect: Rect
    source: str            # uia | ocr
    hwnd: int
    process: str
    title: str
    fg: bool
    visible: bool
    words: tuple = ()      # OCR word boxes ((Rect, word), ...)

    def to_dict(self, rect: Rect | None = None, score: float | None = None) -> dict:
        r = rect or self.rect
        d = {"role": self.role, "name": self.name, "value": self.value, "state": self.state,
             "rect": [r.x, r.y, r.w, r.h], "center": list(r.center), "source": self.source,
             "process": self.process, "title": self.title, "fg": self.fg, "visible": self.visible}
        if score is not None:
            d["score"] = round(score, 3)
        return d


def collect(per, scope: str = "all", area: Rect | None = None, visible_fn=None,
            exclude: list[str] | None = None) -> list[Target]:
    """Every target the perception knows, in screen coordinates. Call under per.lock.

    visible_fn(x, y) -> hwnd of the top-level window at that point; an element hidden
    under another window is not clickable, so it is marked visible=False.
    exclude: skip windows whose title contains any of these (an agent's own log window
    shows the very labels it is looking for).
    """
    out: list[Target] = []
    skip = {h for h, w in per.wins.items() if exclude and any(x and x in w.title for x in exclude)}
    for h, w in per.wins.items():
        if (scope == "fg" and h != per.fg) or h in skip:
            continue
        for e in (w.targets or w.els):
            if not e.rect.area or (area is not None and not area.contains(*e.rect.center)):
                continue
            vis = visible_fn is None or visible_fn(*e.rect.center) == h
            out.append(Target(e.role, e.name, e.value, e.state, e.rect, "uia", h, w.process,
                              w.title, h == per.fg, vis))
    for h, r, t in per.text:
        if (scope == "fg" and h != per.fg) or h in skip:
            continue
        if area is not None and not area.contains(*r.center):
            continue
        w = per.wins.get(h)
        vis = visible_fn is None or visible_fn(*r.center) == h
        out.append(Target("text", t, "", "", r, "ocr", h, w.process if w else "?",
                          w.title if w else "", h == per.fg, vis, per.words.get((h, r, t), ())))
    return out


def _union(rects: list[Rect]) -> Rect:
    x0, y0 = min(r.x for r in rects), min(r.y for r in rects)
    x1, y1 = max(r.x + r.w for r in rects), max(r.y + r.h for r in rects)
    return Rect(x0, y0, x1 - x0, y1 - y0)


def segments(words: tuple) -> list[tuple]:
    """Split an OCR line at wide horizontal gaps: «Mail   Maps   Market» is three labels."""
    if not words:
        return []
    ws = sorted(words, key=lambda w: w[0].x)
    h = max(b.h for b, _ in ws) or 1
    out, cur = [], [ws[0]]
    for w in ws[1:]:
        prev = cur[-1][0]
        if w[0].x - (prev.x + prev.w) > GAP_K * h:
            out.append(tuple(cur))
            cur = [w]
        else:
            cur.append(w)
    out.append(tuple(cur))
    return out


def _ocr_match(query: str, t: Target) -> tuple[float, Rect]:
    """Best run of words in an OCR line: the click goes to those words, not the line.

    Runs never cross a wide gap, and a match inside a long segment is prose (a chat
    message or a paragraph that mentions the label), so it is penalised.
    """
    if not t.words:
        s = similarity(query, t.name)
        return (s - PROSE_PENALTY if len(t.name.split()) > PROSE_WORDS else s), t.rect
    best, rect = 0.0, t.rect
    for seg in segments(t.words):
        penalty = PROSE_PENALTY if len(seg) > PROSE_WORDS else 0.0
        for i in range(len(seg)):
            for j in range(i, min(len(seg), i + MAX_WORDS)):
                run = seg[i: j + 1]
                s = similarity(query, " ".join(w for _, w in run)) - penalty
                if s > best + 1e-9:
                    best, rect = s, _union([b for b, _ in run])
    return best, rect


def score(query: str, t: Target) -> tuple[float, Rect]:
    if t.source == "ocr":
        s, rect = _ocr_match(query, t)
    else:
        s, rect = similarity(query, t.name), t.rect
        if t.value:
            s = max(s, 0.9 * similarity(query, t.value))
    if s <= 0:
        return 0.0, rect
    if t.role in INTERACTIVE:
        s += 0.06
    elif t.role == "text" and t.source == "uia":
        s -= 0.02
    if t.fg:
        s += 0.04
    if not t.visible:
        s -= 0.3
    return s, rect


def find(targets: list[Target], query: str, role: str | None = None, limit: int = 5,
         hidden: bool = False) -> dict:
    """Ranked matches for a label. `ambiguous` warns that several equally good targets exist."""
    hits = []
    for t in targets:
        if role and t.role != role:
            continue
        if not hidden and not t.visible:
            continue
        s, rect = score(query, t)
        if s >= MIN_SCORE:
            hits.append((s, rect, t))
    hits.sort(key=lambda x: -x[0])
    kept: list[tuple[float, Rect, Target]] = []
    for s, rect, t in hits:
        # A link and its inner text, or UIA and OCR reading the same button: one target.
        if any(rect.overlap(k[1]) >= 0.6 * min(rect.area, k[1].area) for k in kept):
            continue
        kept.append((s, rect, t))
        if len(kept) >= limit:
            break
    ambiguous = len(kept) >= 2 and kept[0][0] - kept[1][0] < 0.03 and kept[1][0] >= 0.85
    return {"matches": [t.to_dict(rect, s) for s, rect, t in kept], "ambiguous": ambiguous}


def settle_state(t_req: float, now: float, t_step: float, t_changed: float, dirty: bool,
                 quiet: float, expect: bool) -> str:
    """wait | settled for one poll of /settled.

    A frame grabbed after the request must have been processed (otherwise we would judge
    the screen as it was before the action), the pixels must have been still for `quiet`
    seconds, and changed windows must have been re-read. With expect, a change after the
    request is required first, for actions whose effect shows up with a delay.
    """
    if t_step <= t_req or dirty:
        return "wait"
    if expect and t_changed <= t_req:
        return "wait"
    return "settled" if now - t_changed >= quiet else "wait"
