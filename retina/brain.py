"""Layer 2: the brain. A local Ollama model that keeps the screen log in its KV cache.

Events are appended to the end of one growing prompt, so Ollama evaluates only the new
tokens (prefix cache). A background prefill keeps that cache warm, so a question costs
just its own tokens plus the answer. When the log outgrows its budget, older events are
replaced by a snapshot of the current state.

Note: Ollama's prompt_eval_count reports the whole prompt length even on a cache hit, so
the tokens actually evaluated are tracked as the growth of the prompt between calls.
"""
import threading
import time
from datetime import datetime

import ollama

from .llm import call_chat

SYSTEM = """You see the user's Windows screen through a live text log from a local perception system (UI Automation tree + OCR), not through images.
Log format, each event starts with [hh:mm:ss]:
- "window: proc «title»" - that window came to the foreground; following lines are its visible elements: role «label» = «value» [state].
- "+" appeared, "-" disappeared, "~" changed, "+=" text was typed/appended, "focus →" keyboard focus moved.
- "@proc" - change in a background window (taskbar, notification, another window).
- "▶ motion" / "■ motion stopped" - video or animation playing in a screen region.
- "seen ..." - a vision model's short description of a region with no text (video frame, picture).
- "state snapshot" - the full current picture after older events were dropped.
- «text» without a role - text read from pixels by OCR, may contain recognition errors.
Newest events describe the current screen; window titles often name the open page, video or file.
Answer in the language of the question, in 1-2 sentences unless asked for details. If the log does not show something, say so instead of guessing."""


class Brain:
    def __init__(self, model: str, state_fn, now_fn=None, num_ctx: int = 8192, budget: float = 0.6,
                 debounce: float = 0.3, keep_tail: int = 30):
        self.client = ollama.Client()
        self.model, self.state_fn, self.now_fn, self.num_ctx = model, state_fn, now_fn, num_ctx
        self.budget_tokens = int(num_ctx * budget)
        self.debounce, self.keep_tail = debounce, keep_tail
        self.think = {"think": False} if "qwen3" in model.lower() else {}
        self.log: list[str] = []
        self.log_lock = threading.Lock()
        self.model_lock = threading.Lock()   # one request at a time keeps the cache slot ours
        self.pending = threading.Event()
        self.cached_len = 0       # prompt tokens already in the KV cache (log part)
        self.new_tokens = 0       # prompt tokens actually evaluated, all calls
        self.compactions = 0
        self.error = ""
        threading.Thread(target=self._prefill_loop, daemon=True).start()

    def ingest(self, line: str) -> None:
        with self.log_lock:
            self.log.append(line)
        self.pending.set()

    def ask(self, question: str) -> dict:
        t = time.perf_counter()
        now = self.now_fn() if self.now_fn else ""
        q = f"Now: {now}\nQuestion: {question}" if now else question
        with self.model_lock:
            messages = self._messages(q)
            ttft, parts, last = None, [], None
            for chunk in call_chat(self.client, self.think, model=self.model, messages=messages, stream=True,
                                   keep_alive="30m", options=self._options(200)):
                if ttft is None and chunk.message.content:
                    ttft = time.perf_counter() - t
                parts.append(chunk.message.content)
                last = chunk
            prompt_len = last.prompt_eval_count or 0
            evaluated = max(prompt_len - self.cached_len, 0)
            self.new_tokens += evaluated
        return {"answer": "".join(parts).strip(), "ms": (time.perf_counter() - t) * 1000,
                "ttft_ms": (ttft or 0) * 1000, "prompt_tokens": evaluated, "prompt_len": prompt_len,
                "answer_tokens": last.eval_count or 0}

    def _prefill_loop(self) -> None:
        while True:
            self.pending.wait()
            time.sleep(self.debounce)   # batch bursts of events into one prefill
            self.pending.clear()
            with self.model_lock:
                try:
                    compacted = self._compact_if_needed()
                    r = call_chat(self.client, self.think, model=self.model, messages=self._messages(None),
                                  keep_alive="30m", options=self._options(1))
                    n = r.prompt_eval_count or 0
                    self.new_tokens += n if compacted or n < self.cached_len else n - self.cached_len
                    self.cached_len = n
                    self.error = ""
                except Exception as e:  # Ollama down or model missing: keep watching anyway
                    self.error = str(e)

    def _messages(self, question: str | None) -> list[dict]:
        with self.log_lock:
            log = "\n".join(self.log)
        messages = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": "Screen log:\n" + log}]
        if question:
            messages.append({"role": "user", "content": question})
        return messages

    def _options(self, num_predict: int) -> dict:
        return {"num_ctx": self.num_ctx, "num_predict": num_predict, "temperature": 0.2}

    def _compact_if_needed(self) -> bool:
        with self.log_lock:
            # cached_len lags behind the newest events; ~3 chars per token is a safe upper estimate.
            est = max(self.cached_len, sum(len(x) + 1 for x in self.log) // 3 + len(SYSTEM) // 3)
            if est <= self.budget_tokens:
                return False
            # Leave half the budget free after compacting, or the next prefill compacts again.
            tail, size = [], 0
            for line in reversed(self.log[-self.keep_tail:]):
                size += len(line) + 1
                if size > self.budget_tokens * 3 * 0.15:
                    break
                tail.insert(0, line)
        state = self.state_fn()[: int(self.budget_tokens * 3 * 0.35)]
        snapshot = f"[{datetime.now():%H:%M:%S}] state snapshot:\n{state}"
        with self.log_lock:
            self.log = tail + [snapshot]
            self.compactions += 1
            self.cached_len = 0
        return True
