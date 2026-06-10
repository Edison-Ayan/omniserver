"""Core data structures for the inference engine.

A `Request` is what the API layer submits. Internally the engine tracks each in
flight request as a `Sequence` that carries its token state, KV-cache handle and
generation status across scheduler iterations (continuous batching means a
sequence lives across many model steps, being admitted, run, and retired
independently of other sequences in the batch).
"""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

_counter = itertools.count()


class Status(enum.Enum):
    WAITING = "waiting"      # in the queue, prefill not started
    RUNNING = "running"      # actively decoding in the batch
    FINISHED = "finished"    # hit EOS or max tokens
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0     # 0 -> greedy
    top_p: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Request:
    """User-facing unit submitted to the engine."""
    prompt: str
    images: List[Image.Image] = field(default_factory=list)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{next(_counter)}")
    arrival_t: float = field(default_factory=time.perf_counter)


@dataclass
class Sequence:
    """Engine-internal mutable state for one in-flight request."""
    request: Request
    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)
    status: Status = Status.WAITING

    # Filled by the model runner during prefill; carried across decode steps.
    # Kept as an opaque handle so we can swap contiguous KV today for paged KV
    # later without touching the scheduler.
    kv_handle: Optional[object] = None
    # Vision token embeddings produced (or cache-served) at prefill.
    vision_embeds: Optional[object] = None

    # Streaming bookkeeping: how many output tokens have been sent to the client.
    num_streamed: int = 0

    # Timestamps for metrics.
    first_token_t: Optional[float] = None
    finish_t: Optional[float] = None

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        if self.first_token_t is None:
            self.first_token_t = time.perf_counter()
        self.output_token_ids.append(token_id)

    def maybe_finish(self) -> bool:
        sp = self.request.sampling
        if self.output_token_ids and self.output_token_ids[-1] in sp.stop_token_ids:
            self._finish()
            return True
        if self.num_output_tokens >= sp.max_new_tokens:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        self.status = Status.FINISHED
        self.finish_t = time.perf_counter()
