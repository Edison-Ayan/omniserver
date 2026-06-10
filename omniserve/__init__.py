"""omniserve — a lightweight multimodal LLM inference engine.

Public surface kept small on purpose; the interesting parts are the scheduler
(continuous batching) and the model runner (KV cache + vision embedding cache).
"""

from .engine import LLMEngine, StepDelta
from .model_runner import ModelRunner, StubRunner
from .request import Request, SamplingParams, Sequence, Status
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
        from .qwen_runner import QwenVLRunner
        return QwenVLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
