"""omniserve —— 一个轻量的多模态 LLM 推理引擎。

对外暴露的接口刻意保持精简;有意思的部分是调度器(continuous batching)和
model runner(KV cache + vision embedding cache)。
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
    # 懒加载,这样没装 torch/transformers 也能 import 控制平面。
    if name == "QwenVLRunner":
        from .runners.qwen_vl import QwenVLRunner
        return QwenVLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
