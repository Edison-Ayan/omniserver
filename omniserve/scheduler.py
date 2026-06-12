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
    max_prefill_per_step: int = 1  # 每次迭代 prefill 多少个新序列
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
        #    prefill 是多模态里最贵的一步,所以限制每次迭代准入的数量,
        #    避免给已经在 decode 的序列带来延迟尖峰。
        admitted = 0
        while (
            self.waiting
            and len(self.running) < self.config.max_running
            and admitted < self.config.max_prefill_per_step
        ):
            seq = self.waiting.popleft()
            if seq.status == Status.ABORTED:
                continue
            seq.status = Status.RUNNING
            self.running.append(seq)
            out.prefill.append(seq)
            admitted += 1

        # 3) 其余每个 running 序列前进一个 decode token。
        out.decode = [s for s in self.running if s not in out.prefill]
        return out

    def free_finished(self) -> List[Sequence]:
        """返回本步跑完的序列,好让引擎释放它们的 KV cache 并通知客户端。"""
        done = [s for s in self.running if s.status != Status.RUNNING]
        self.running = [s for s in self.running if s.status == Status.RUNNING]
        return done
