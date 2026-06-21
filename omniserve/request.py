"""推理引擎的核心数据结构。

`Request` 是 API 层提交的请求。引擎内部把每个在途请求追踪为一个 `Sequence`,它在
调度器的多次迭代间携带 token 状态、KV-cache 句柄和生成状态(continuous batching
意味着一个序列会跨越很多个模型步:被准入、运行、退休,和 batch 里其他序列相互独立)。
"""

from __future__ import annotations

import enum
import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional

from PIL import Image

_counter = itertools.count()


class Status(enum.Enum):
    WAITING = "waiting"      # 在队列里,还没开始 prefill
    RUNNING = "running"      # 正在 batch 里 decode
    FINISHED = "finished"    # 命中 EOS 或达到 max tokens
    ABORTED = "aborted"


@dataclass
class SamplingParams:
    max_new_tokens: int = 128
    temperature: float = 0.0     # 0 表示 greedy
    top_p: float = 1.0
    stop_token_ids: List[int] = field(default_factory=list)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0


@dataclass
class Request:
    """提交给引擎的用户层请求单元。"""
    prompt: str
    images: List[Image.Image] = field(default_factory=list)
    sampling: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = field(default_factory=lambda: f"req-{next(_counter)}")
    arrival_t: float = field(default_factory=time.perf_counter)


@dataclass
class Sequence:
    """单个在途请求的引擎内部可变状态。"""
    request: Request
    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)
    status: Status = Status.WAITING

    # 由 model runner 在 prefill 时填充,跨 decode 步携带。
    # 存成不透明句柄,这样以后从连续 KV 换成分页 KV 时不用动调度器。
    kv_handle: Optional[object] = None
    # prefill 时产生(或 cache 命中拿到)的 vision token embedding。
    vision_embeds: Optional[object] = None

    # chunked prefill:已 prefill 的 prompt token 数(< num_prompt_tokens 表示还在分块 prefill 中)。
    # 整段一次 prefill 的序列在准入时就被置成 num_prompt_tokens;长 prompt 逐块累加。
    num_prefilled: int = 0
    # 调度器写、runner 读:这一步要 prefill 多少个 token(分块 prefill 用)。
    prefill_chunk: int = 0

    # 流式记账:已经发给客户端多少个 output token。
    num_streamed: int = 0

    # 指标用的时间戳。
    first_token_t: Optional[float] = None
    finish_t: Optional[float] = None

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_len(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    def last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token(self, token_id: int) -> None:
        if self.first_token_t is None:
            self.first_token_t = time.perf_counter()
        self.output_token_ids.append(token_id)

    def maybe_finish(self) -> bool:
        sp = self.request.sampling
        if self.output_token_ids and self.output_token_ids[-1] in sp.stop_token_ids:
            self._finish()
            return True
        if self.num_output_tokens >= sp.max_new_tokens:
            self._finish()
            return True
        return False

    def _finish(self) -> None:
        self.status = Status.FINISHED
        self.finish_t = time.perf_counter()
