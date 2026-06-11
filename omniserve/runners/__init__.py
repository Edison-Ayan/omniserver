"""Model execution backends for the engine.

`base` defines the `ModelRunner` interface and a no-GPU `StubRunner`.
Concrete model runners (e.g. `QwenVLRunner`) live alongside it and are imported
lazily so the control plane stays importable without torch/transformers.
"""

from .base import ModelRunner, StubRunner

__all__ = ["ModelRunner", "StubRunner", "QwenVLRunner"]


def __getattr__(name):
    if name == "QwenVLRunner":
        from .qwen_vl import QwenVLRunner
        return QwenVLRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
