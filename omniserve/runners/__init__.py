"""引擎的模型执行后端。

`base` 定义了 `ModelRunner` 接口和无 GPU 的 `StubRunner`。
具体的 model runner(比如 `QwenVLRunner`)放在旁边,并且懒加载,
这样没装 torch/transformers 也能 import 控制平面。
"""

from .base import ModelRunner, StubRunner

__all__ = ["ModelRunner", "StubRunner", "QwenVLRunner"]


def __getattr__(name):
    if name == "QwenVLRunner":
        from .qwen_vl import QwenVLRunner
        return QwenVLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
