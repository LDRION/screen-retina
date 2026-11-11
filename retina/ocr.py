"""Layer 1a: built-in Windows OCR, used only where UI Automation has no text."""
from __future__ import annotations

import asyncio
import hashlib
from collections import OrderedDict
from typing import NamedTuple

import numpy as np
from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import BitmapAlphaMode, BitmapPixelFormat, SoftwareBitmap
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter

from .geom import Rect


class Ocr:
    def __init__(self, lang: str | None = None, upscale_below: int = 400, cache_size: int = 256):
        self.engine = (OcrEngine.try_create_from_language(Language(lang)) if lang
                       else OcrEngine.try_create_from_user_profile_languages())
        if self.engine is None:
            raise RuntimeError(f"no Windows OCR engine for {lang or 'user profile languages'}")
        self.lang = self.engine.recognizer_language.language_tag
        self.upscale_below = upscale_below
        self.loop = asyncio.new_event_loop()
        self.cache: OrderedDict[bytes, list["Line"]] = OrderedDict()
        self.cache_size = cache_size

    def read(self, crop: np.ndarray, r: Rect) -> list[Line]:
        """OCR the pixels of screen rect r; returns text lines with their screen rects.

        Each line also carries its word boxes: one OCR line can glue neighbouring menu
        items («Mail Maps Market»), and a click must land on the word, not the line centre.
        """
        crop = np.ascontiguousarray(crop)
        if crop.size == 0:
            return []
        key = hashlib.blake2b(crop.tobytes(), digest_size=16).digest()
        if key in self.cache:
            self.cache.move_to_end(key)
            return [_shift(ln, r.x, r.y) for ln in self.cache[key]]

        # Small regions read better upscaled; plain pixel repeat is enough for screen text.
        k = 2 if min(crop.shape[:2]) < self.upscale_below else 1
        img = crop.repeat(k, 0).repeat(k, 1) if k > 1 else crop
        lines = self.loop.run_until_complete(self._recognize(img))
        local = [_scale(ln, k) for ln in lines]
        self.cache[key] = local
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return [_shift(ln, r.x, r.y) for ln in local]

    async def _recognize(self, img: np.ndarray) -> list[Line]:
        h, w = img.shape[:2]
        writer = DataWriter()
        writer.write_bytes(img.tobytes())
        bmp = SoftwareBitmap.create_copy_with_alpha_from_buffer(
            writer.detach_buffer(), BitmapPixelFormat.BGRA8, w, h, BitmapAlphaMode.PREMULTIPLIED)
        result = await self.engine.recognize_async(bmp)
        out = []
        for line in result.lines:
            words = tuple((Rect(int(wd.bounding_rect.x), int(wd.bounding_rect.y),
                                int(wd.bounding_rect.width), int(wd.bounding_rect.height)), wd.text)
                          for wd in line.words)
            if not words:
                continue
            x0, y0 = min(b.x for b, _ in words), min(b.y for b, _ in words)
            x1, y1 = max(b.x + b.w for b, _ in words), max(b.y + b.h for b, _ in words)
            out.append(Line(Rect(x0, y0, x1 - x0, y1 - y0), line.text, words))
        return out


class Line(NamedTuple):
    rect: Rect
    text: str
    words: tuple = ()     # ((Rect, word), ...) in the same coordinates as rect


def _shift(ln: Line, dx: int, dy: int) -> Line:
    return Line(Rect(ln.rect.x + dx, ln.rect.y + dy, ln.rect.w, ln.rect.h), ln.text,
                tuple((Rect(b.x + dx, b.y + dy, b.w, b.h), t) for b, t in ln.words))


def _scale(ln: Line, k: int) -> Line:
    if k == 1:
        return ln
    return Line(Rect(ln.rect.x // k, ln.rect.y // k, ln.rect.w // k, ln.rect.h // k), ln.text,
                tuple((Rect(b.x // k, b.y // k, b.w // k, b.h // k), t) for b, t in ln.words))
