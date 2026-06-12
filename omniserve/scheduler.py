"""Continuous-batching(连续批处理)调度器。

不同于静态批处理(等齐 N 个请求、一起跑到全部结束),continuous batching 在**每个
模型步**都做一次调度决策:跑完的序列立刻离开 batch,等待中的序列补进空出来的槽位。
这是相比朴素 `model.generate` 循环最重要的吞吐提升点,也是面试常考的地方。

这个调度器刻意做成与后端无关:它只决定**哪些**序列接下来跑、跑哪个阶段;具体**怎么**
执行由 model runner 决定。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List

from .request import Sequence, Status


@dataclass
class SchedulerConfig:
    max_running: int = 8          # 同时 decode 的最大序列数(batch 上限)
    max_prefill_per_step: int = 1  # 每步最多准入几个新序列(打包式 prefill 时调大)
    max_prefill_tokens: int = 8192  # 每步 prefill 的 token 预算(打包多少个 prompt 进一次前向)
    max_seq_len: int = 4096        # prompt+output 长度的硬上限


@dataclass
class SchedulerOutput:
    """本次迭代引擎该执行的内容。"""
    prefill: List[Sequence] = field(default_factory=list)  # 需要做 prefill
    decode: List[Sequence] = field(default_factory=list)    # 前进一个 token


class Scheduler:
    def __init__(self, config: SchedulerConfig):
        self.config = config
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def abort(self, request_id: str) -> None:
        for q in (self.waiting, self.running):
            for seq in list(q):
                if seq.request_id == request_id:
                    seq.status = Status.ABORTED
                    if seq in self.waiting:
                        self.waiting.remove(seq)

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()

        # 1) 把跑完/被中止的序列从 running 集合里退休掉(腾出槽位)。
        self.running = [s for s in self.running if s.status == Status.RUNNING]

        # 2) 在还有 batch 容量时准入等待中的序列(prefill)。
        #    打包式 prefill:一步可以准入多个 prompt 打进一次前向,受两个上限约束——
        #    数量上限 max_prefill_per_step,和 token 预算 max_prefill_tokens(总 prompt
        #    token 数不超预算;至少准入一个,避免超长 prompt 永远进不来)。
        admitted = 0
        used_tokens = 0
        while (
            self.waiting
            and len(self.running) < self.config.max_running
            and admitted < self.config.max_prefill_per_step
        ):
            seq = self.waiting[0]
            n_tok = seq.num_prompt_tokens
            # 已经打包了至少一个、再加这个会超预算 -> 这步先不收
            if admitted > 0 and used_tokens + n_tok > self.config.max_prefill_tokens:
                break
            self.waiting.popleft()
            if seq.status == Status.ABORTED:
                continue
            seq.status = Status.RUNNING
            self.running.append(seq)
            out.prefill.append(seq)
            admitted += 1
            used_tokens += n_tok

        # 3) 其余每个 running 序列前进一个 decode token。
        out.decode = [s for s in self.running if s not in out.prefill]
        return out

    def free_finished(self) -> List[Sequence]:
        """返回本步跑完的序列,好让引擎释放它们的 KV cache 并通知客户端。"""
        done = [s for s in self.running if s.status != Status.RUNNING]
        self.running = [s for s in self.running if s.status == Status.RUNNING]
        return done
