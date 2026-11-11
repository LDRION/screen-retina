"""Layer 1b: eyes. A local vision model describes regions with no structure and no text
(video frames, pictures). It runs in the background and only on demand: when motion starts,
when the scene inside a moving region changes, or when a large new image appears.
"""
import io
import queue
import threading

import numpy as np
import ollama
from PIL import Image

from .geom import Rect
from .llm import call_chat

PROMPT = ("This is a crop of a computer screen. In one sentence of at most 20 words, say what it shows "
          "(scene, people, objects, large text). No preamble.")


def rgb(crop: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(crop[..., 2::-1])   # BGRA -> RGB


class Eyes:
    def __init__(self, model: str, num_ctx: int = 8192, max_side: int = 640):
        self.client = ollama.Client()
        self.model, self.num_ctx, self.max_side = model, num_ctx, max_side
        self.think = {"think": False} if "qwen3" in model.lower() else {}
        self.inbox: queue.Queue = queue.Queue(maxsize=2)
        self.out: queue.Queue = queue.Queue()
        self.looks = 0
        self.image_tokens = 0
        self.error = ""
        threading.Thread(target=self._loop, daemon=True).start()

    def look(self, crop: np.ndarray, r: Rect, label: str, t_frame: float) -> None:
        """Queue a BGRA crop of screen rect r for a background description."""
        try:
            self.inbox.put_nowait((rgb(crop), r, label, t_frame))
        except queue.Full:
            pass   # still busy with earlier looks; a newer request will follow if it matters

    def describe(self, crop: np.ndarray) -> tuple[str, int]:
        """Caption one RGB crop synchronously. Returns (text, image prompt tokens)."""
        img = Image.fromarray(crop)
        img.thumbnail((self.max_side, self.max_side))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        resp = call_chat(self.client, self.think, model=self.model, keep_alive="30m",
                         messages=[{"role": "user", "content": PROMPT, "images": [buf.getvalue()]}],
                         options={"num_ctx": self.num_ctx, "num_predict": 60, "temperature": 0.2})
        self.looks += 1
        self.image_tokens += resp.prompt_eval_count or 0
        return " ".join(resp.message.content.split()), resp.prompt_eval_count or 0

    def _loop(self) -> None:
        while True:
            crop, r, label, t_frame = self.inbox.get()
            try:
                text, _ = self.describe(crop)
                self.out.put((t_frame, f"seen {label} {r}: {text}"))
                self.error = ""
            except Exception as e:   # model missing or Ollama down: perception keeps working
                self.error = str(e)
