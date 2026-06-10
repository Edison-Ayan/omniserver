"""The inference engine: drives the scheduler/runner step loop.

One `step()` is one iteration of continuous batching:
    schedule -> prefill new seqs + decode running seqs -> retire finished.

Per-step it returns the increments produced by each sequence so the server can
stream tokens to clients as soon as they are generated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .model_runner import ModelRunner
from .request import Request, Sequence, Status
from .scheduler import Scheduler, SchedulerConfig


@dataclass
class StepDelta:
    """Per-sequence output produced in one step, for streaming back to clients."""
    request_id: str
    text: str
    finished: bool


class LLMEngine:
    def __init__(self, runner: ModelRunner, sched_config: SchedulerConfig | None = None):
        self.runner = runner
        self.scheduler = Scheduler(sched_config or SchedulerConfig())
        self._seqs: Dict[str, Sequence] = {}

    def add_request(self, request: Request) -> str:
        seq = Sequence(request=request)
        self.runner.tokenize(seq)
        self._seqs[seq.request_id] = seq
        self.scheduler.add(seq)
        return seq.request_id

    def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> List[StepDelta]:
        out = self.scheduler.schedule()
        deltas: List[StepDelta] = []

        if out.prefill:
            self.runner.prefill(out.prefill)
        if out.decode:
            self.runner.decode(out.decode)

        # Collect newly streamed tokens from every sequence touched this step.
        for seq in (*out.prefill, *out.decode):
            new_ids = seq.output_token_ids[seq.num_streamed:]
            if new_ids:
                text = self.runner.detokenize(seq, new_ids)
                seq.num_streamed = seq.num_output_tokens
                deltas.append(StepDelta(seq.request_id, text, seq.status != Status.RUNNING))

        # Release finished sequences (KV cache freeing happens in the runner via
        # the kv_handle going out of scope / explicit free hook later).
        for seq in self.scheduler.free_finished():
            self._seqs.pop(seq.request_id, None)

        return deltas

    def run_until_complete(self) -> Dict[str, str]:
        """Synchronous helper for offline/batch use and tests."""
        texts: Dict[str, List[str]] = {rid: [] for rid in self._seqs}
        while self.has_unfinished():
            for d in self.step():
                texts.setdefault(d.request_id, []).append(d.text)
        return {rid: "".join(parts) for rid, parts in texts.items()}
