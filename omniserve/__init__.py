"""omniserve — a lightweight multimodal LLM inference engine.

Public surface kept small on purpose; the interesting parts are the scheduler
(continuous batching) and the model runner (KV cache + vision embedding cache).
"""

from .engine import LLMEngine, StepDelta
from .request import Request, SamplingParams, Sequence, Status
from .runners import ModelRunner, StubRunner
from .scheduler import Scheduler, SchedulerConfig

__all__ = [
    "LLMEngine",
    "StepDelta",
    "ModelRunner",
    "StubRunner",
    "Request",
    "SamplingParams",
    "Sequence",
    "Status",
    "Scheduler",
    "SchedulerConfig",
]


def __getattr__(name):
    # Lazy import so the control plane stays importable without torch/transformers.
    if name == "QwenVLRunner":
        from .runners.qwen_vl import QwenVLRunner
        return QwenVLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
