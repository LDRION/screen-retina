"""Rectangles and the coarse cell grid shared by all layers."""
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def overlap(self, o: "Rect") -> int:
        dx = min(self.x + self.w, o.x + o.w) - max(self.x, o.x)
        dy = min(self.y + self.h, o.y + o.h) - max(self.y, o.y)
        return dx * dy if dx > 0 and dy > 0 else 0

    def contains(self, px: int, py: int) -> bool:
        return self.x <= px < self.x + self.w and self.y <= py < self.y + self.h

    def clip(self, o: "Rect") -> "Rect | None":
        x0, y0 = max(self.x, o.x), max(self.y, o.y)
        x1, y1 = min(self.x + self.w, o.x + o.w), min(self.y + self.h, o.y + o.h)
        return Rect(x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None

    def __str__(self) -> str:
        return f"{self.x},{self.y} {self.w}x{self.h}"


class Grid:
    """Virtual screen split into square cells; masks are bool arrays of shape (rows, cols).

    All rects are in screen coordinates; (ox, oy) is the top-left of the virtual screen,
    negative when a monitor sits left of or above the primary one.
    """

    def __init__(self, width: int, height: int, cell: int, ox: int = 0, oy: int = 0):
        self.width, self.height, self.cell = width, height, cell
        self.ox, self.oy = ox, oy
        self.rows = -(-height // cell)
        self.cols = -(-width // cell)

    @property
    def bounds(self) -> Rect:
        return Rect(self.ox, self.oy, self.width, self.height)

    def cell_center(self, row: int, col: int) -> tuple[int, int]:
        c = self.cell
        return int(self.ox + col * c + c // 2), int(self.oy + row * c + c // 2)

    def cell_at(self, x: int, y: int) -> tuple[int, int] | None:
        row, col = (y - self.oy) // self.cell, (x - self.ox) // self.cell
        return (row, col) if 0 <= row < self.rows and 0 <= col < self.cols else None

    def crop(self, frame: np.ndarray, r: Rect) -> tuple[np.ndarray, Rect | None]:
        """Pixels of a screen rect (clipped to the screen) and the rect actually cut."""
        r = r.clip(self.bounds)
        if r is None:
            return frame[:0, :0], None
        x, y = r.x - self.ox, r.y - self.oy
        return frame[y: y + r.h, x: x + r.w], r

    def empty(self) -> np.ndarray:
        return np.zeros((self.rows, self.cols), bool)

    def components(self, mask: np.ndarray) -> list[Rect]:
        """Bounding boxes (in pixels) of 8-connected groups of True cells."""
        seen = np.zeros_like(mask)
        out = []
        for r0, c0 in zip(*np.nonzero(mask)):
            if seen[r0, c0]:
                continue
            seen[r0, c0] = True
            q = deque([(r0, c0)])
            rmin = rmax = r0
            cmin = cmax = c0
            while q:
                r, c = q.popleft()
                rmin, rmax, cmin, cmax = min(rmin, r), max(rmax, r), min(cmin, c), max(cmax, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < self.rows and 0 <= cc < self.cols and mask[rr, cc] and not seen[rr, cc]:
                            seen[rr, cc] = True
                            q.append((rr, cc))
            x, y = cmin * self.cell, rmin * self.cell
            w = min((cmax + 1) * self.cell, self.width) - x
            h = min((rmax + 1) * self.cell, self.height) - y
            out.append(Rect(int(x + self.ox), int(y + self.oy), int(w), int(h)))
        return out
