"""Layer 0a: pixels. Grabs the screen and reports which grid cells changed.

Cells that keep changing (video, animation) become "hot" and are reported separately,
so the expensive layers skip them. When a hot area calms down it is reported once as
"settled", so its final content still gets read.
"""
import time

import mss
import numpy as np

from .geom import Grid


class PixelDiff:
    def __init__(self, cell: int = 32, scale: int = 4, threshold: int = 18,
                 heat_decay: float = 0.85, hot_level: float = 4.0, monitor: int = 0):
        self.sct = mss.MSS()
        self.mon = self.sct.monitors[monitor]
        # monitor 0 is the whole virtual screen: every monitor at once.
        self.grid = Grid(self.mon["width"], self.mon["height"], cell, self.mon["left"], self.mon["top"])
        self.scale = scale
        self.threshold = threshold
        self.heat_decay = heat_decay
        self.hot_level = hot_level
        self.prev = None
        self.heat = np.zeros((self.grid.rows, self.grid.cols), np.float32)
        self.was_hot = self.grid.empty()

    def grab(self) -> tuple[np.ndarray, float]:
        t = time.perf_counter()
        return np.asarray(self.sct.grab(self.mon)), t

    def diff(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (changed, hot) cell masks. `changed` excludes hot cells but includes settled ones."""
        g = frame[:: self.scale, :: self.scale, 1].astype(np.int16)
        if self.prev is None or self.prev.shape != g.shape:
            self.prev = g
            return ~self.grid.empty(), self.grid.empty()
        moved = np.abs(g - self.prev) > self.threshold
        self.prev = g

        c = self.grid.cell // self.scale
        pad = np.zeros((self.grid.rows * c, self.grid.cols * c), bool)
        pad[: moved.shape[0], : moved.shape[1]] = moved
        cells = pad.reshape(self.grid.rows, c, self.grid.cols, c).any(axis=(1, 3))

        self.heat = self.heat * self.heat_decay + cells
        hot = self.heat >= self.hot_level
        settled = self.was_hot & ~hot
        self.was_hot = hot
        return (cells & ~hot) | settled, hot
