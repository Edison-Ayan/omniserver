"""Continuous-batching scheduler.

Unlike static batching (wait for N requests, run them to completion together),
continuous batching makes a scheduling decision *every model step*: finished
sequences leave the batch immediately and waiting sequences are admitted to fill
the freed slots. This is the single most important throughput win over a naive
`model.generate` loop, and the piece interviewers probe on.

This scheduler is deliberately backend-agnostic: it decides *which* sequences
run next and in what phase; the model runner decides *how* to execute them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

from .request import Sequence, Status


@dataclass
class SchedulerConfig:
    max_running: int = 8          # max sequences decoding concurrently (batch cap)
    max_prefill_per_step: int = 1  # how many new seqs to prefill per iteration
    max_seq_len: int = 4096        # hard cap on prompt+output length


@dataclass
class SchedulerOutput:
    """What the engine should execute this iteration."""
    prefill: List[Sequence] = field(default_factory=list)  # need a prefill pass
    decode: List[Sequence] = field(default_factory=list)    # advance one token


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> None:
        for q in (self.waiting, self.running):
            for seq in list(q):
                if seq.request_id == request_id:
                    seq.status = Status.ABORTED
                    if seq in self.waiting:
                        self.waiting.remove(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()

        # 1) Retire finished/aborted sequences from the running set (free slots).
        self.running = [s for s in self.running if s.status == Status.RUNNING]

        # 2) Admit waiting sequences (prefill) while there is batch capacity.
        #    Prefill is the expensive multimodal step, so we cap how many we do
        #    per iteration to avoid a latency spike for already-running decodes.
        admitted = 0
        while (
            self.waiting
            and len(self.running) < self.config.max_running
            and admitted < self.config.max_prefill_per_step
        ):
            seq = self.waiting.popleft()
            if seq.status == Status.ABORTED:
                continue
            seq.status = Status.RUNNING
            self.running.append(seq)
            out.prefill.append(seq)
            admitted += 1

        # 3) Every other running sequence advances by one decode token.
        out.decode = [s for s in self.running if s not in out.prefill]
        return out

    def free_finished(self) -> List[Sequence]:
        """Return sequences that finished this step so the engine can release
        their KV cache and notify the client."""
        done = [s for s in self.running if s.status != Status.RUNNING]
        self.running = [s for s in self.running if s.status == Status.RUNNING]
        return done
