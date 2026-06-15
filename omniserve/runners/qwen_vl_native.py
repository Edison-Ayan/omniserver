"""Qwen2-VL 的便捷入口(向后兼容)。

解耦后:模型逻辑全在 models/qwen2_vl.py 的 `Qwen2VLAdapter`,调度/KV/graph 全在
runners/multimodal.py 的 `MultimodalRunner`。本类只是把两者拼起来的薄封装,保留旧的
`NativeQwenVLRunner` 名字和构造签名,这样 run_omniserve / server 等老代码不用改。
"""

from __future__ import annotations

from ..models.qwen2_vl import Qwen2VLAdapter
from .multimodal import MultimodalRunner


class NativeQwenVLRunner(MultimodalRunner):
    def __init__(self, model_dir: str = None, max_running: int = 32, max_len: int = 1024,
                 prefix_cache=None, quant: str = None, gptq_dir: str = None,
                 use_graph: bool = False):
        adapter = Qwen2VLAdapter(model_dir=model_dir, quant=quant, gptq_dir=gptq_dir)
        super().__init__(adapter, max_running=max_running, max_len=max_len,
                         prefix_cache=prefix_cache, use_graph=use_graph)
