"""VLMAdapter 适配层:把"模型怎么算"和"引擎怎么调度"解耦。加新模型 = 加一个 adapter。"""

from .base import VLMAdapter
from .qwen2_vl import Qwen2VLAdapter

__all__ = ["VLMAdapter", "Qwen2VLAdapter"]
