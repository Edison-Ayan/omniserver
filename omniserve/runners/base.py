"""模型执行后端。

runner 是唯一碰张量 / GPU 的组件。它给引擎暴露两个操作:

    prefill(seqs)  -> 对每个新序列,编码 prompt(文本+图像),跑一次模型,
                      填好它的 KV cache,吐出第一个 token。
    decode(seqs)   -> 对每个在跑的序列,用它缓存的 KV 状态做单 token 前向,
                      吐出下一个 token。

`ModelRunner` 是抽象接口,让引擎与后端无关。`QwenVLRunner` 是具体的 Qwen2-VL 实现。
vision embedding cache 在 prefill 时接入(图 -> vision token),是本项目的差异化优化。

注意:Qwen2-VL 的 KV-cache 接线(每序列一个 DynamicCache、左 padding 的批量 decode)
是必须在 GPU 上验证的部分。这里先按清晰的契约搭好;上面的引擎和调度器已经完整,
可以用 StubRunner 测试。
"""

from __future__ import annotations

import abc
from typing import List

from ..request import Sequence


class ModelRunner(abc.ABC):
    @abc.abstractmethod
    def tokenize(self, seq: Sequence) -> None:
        """从请求 prompt(+ 图像占位符)填好 seq.prompt_token_ids。"""

    @abc.abstractmethod
    def prefill(self, seqs: List[Sequence]) -> None:
        """对每个 seq 做 prompt prefill;追加第一个生成 token,并在序列上存好 KV 句柄。"""

    @abc.abstractmethod
    def decode(self, seqs: List[Sequence]) -> None:
        """用每个在跑序列的 KV 句柄,把它前进恰好一个 token。"""

    @abc.abstractmethod
    def detokenize(self, seq: Sequence, new_token_ids: List[int]) -> str:
        """把新产生的 token id 渲染成文本,用于流式输出。"""

    def free(self, seq: Sequence) -> None:
        """释放跑完序列占用的资源(比如一个 KV-cache 槽位)。默认空操作;
        有预分配池的后端会重写它。"""


class StubRunner(ModelRunner):
    """无 GPU 的 runner,让我们能端到端地跑通调度器/引擎。

    吐固定的、确定性的假回复,这样在真模型接上之前就能测控制平面
    (batching、流式、退休)。
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
