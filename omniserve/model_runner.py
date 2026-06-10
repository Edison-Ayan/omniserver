"""Model execution backend.

The runner is the only component that touches tensors / the GPU. It exposes two
operations to the engine:

    prefill(seqs)  -> for each new sequence, encode the prompt (text + images),
                      run the model once, populate its KV cache, emit 1st token.
    decode(seqs)   -> for each running sequence, run a single-token forward using
                      its cached KV state, emit the next token.

`ModelRunner` is the abstract interface so the engine stays backend-agnostic.
`QwenVLRunner` is the concrete Qwen2-VL implementation. The vision embedding
cache plugs in at prefill time (image -> vision tokens) and is the project's
differentiating optimization.

NOTE: the Qwen2-VL KV-cache wiring (DynamicCache per sequence, batched decode
with left-padding) is the part that must be validated on the GPU. It is staged
here with a clear contract; the engine and scheduler above are already complete
and testable against the StubRunner.
"""

from __future__ import annotations

import abc
from typing import List

from .request import Sequence


class ModelRunner(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, seq: Sequence) -> None:
        """Fill seq.prompt_token_ids from the request prompt (+ image placeholders)."""

    @abc.abstractmethod
    def prefill(self, seqs: List[Sequence]) -> None:
        """Run prompt prefill for each seq; append the first generated token and
        store the KV handle on the sequence."""

    @abc.abstractmethod
    def decode(self, seqs: List[Sequence]) -> None:
        """Advance each running seq by exactly one token using its KV handle."""

    @abc.abstractmethod
    def detokenize(self, seq: Sequence, new_token_ids: List[int]) -> str:
        """Render newly produced token ids to text for streaming."""


class StubRunner(ModelRunner):
    """No-GPU runner that lets us exercise the scheduler/engine end to end.

    Emits a deterministic canned response so the control plane (batching,
    streaming, retirement) can be tested before the real model is wired in.
    """

    def __init__(self, reply: str = "ok " * 8):
        self._reply_tokens = reply.split()

    def tokenize(self, seq: Sequence) -> None:
        seq.prompt_token_ids = list(range(max(1, len(seq.request.prompt.split()))))

    def prefill(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            seq.kv_handle = {"len": seq.num_prompt_tokens}
            seq.append_token(0)
            seq.maybe_finish()

    def decode(self, seqs: List[Sequence]) -> None:
        for seq in seqs:
            idx = seq.num_output_tokens
            seq.append_token(idx)
            seq.kv_handle["len"] += 1
            seq.maybe_finish()

    def detokenize(self, seq: Sequence, new_token_ids: List[int]) -> str:
        parts = []
        for tid in new_token_ids:
            i = tid % len(self._reply_tokens)
            parts.append(self._reply_tokens[i])
        return " ".join(parts) + " "
