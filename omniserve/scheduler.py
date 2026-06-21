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
    prefill: List[Sequence] = field(default_factory=list)  # 整段一次 prefill(短 prompt,可打包)
    decode: List[Sequence] = field(default_factory=list)    # 前进一个 token
    prefill_chunk: "Sequence | None" = None  # 分块 prefill 中的长 prompt(这一步算一块)


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
        budget = self.config.max_prefill_tokens

        # 1) 把跑完/被中止的序列从 running 集合里退休掉(腾出槽位)。
        self.running = [s for s in self.running if s.status == Status.RUNNING]

        # 2) chunked prefill 优先:若有长 prompt 正在分块 prefill(num_prefilled 没满),
        #    这一步继续算它的下一块(每步最多 budget 个 token),decode 照常进行不被阻塞。
        inflight = next(
            (s for s in self.running if s.num_prefilled < s.num_prompt_tokens), None)
        if inflight is not None:
            inflight.prefill_chunk = min(
                inflight.num_prompt_tokens - inflight.num_prefilled, budget)
            out.prefill_chunk = inflight
        else:
            # 3) 准入等待中的序列做整段 prefill。受数量上限 max_prefill_per_step 和 token 预算约束。
            #    例外:单个 prompt 比预算还长 -> 走分块 prefill(独占这一步,不和别人打包),
            #    这样它不会一次算完整段把正在跑的 decode 堵住。
            admitted = 0
            used_tokens = 0
            while (
                self.waiting
                and len(self.running) < self.config.max_running
                and admitted < self.config.max_prefill_per_step
            ):
                seq = self.waiting[0]
                n_tok = seq.num_prompt_tokens
                if n_tok > budget:                       # 长 prompt -> 分块路径
                    if admitted > 0:
                        break                            # 先把本步打包的短 prompt 跑完,下步再开它
                    self.waiting.popleft()
                    if seq.status == Status.ABORTED:
                        continue
                    seq.status = Status.RUNNING
                    self.running.append(seq)
                    seq.prefill_chunk = min(n_tok, budget)
                    out.prefill_chunk = seq
                    break
                # 已经打包了至少一个、再加这个会超预算 -> 这步先不收
                if admitted > 0 and used_tokens + n_tok > budget:
                    break
                self.waiting.popleft()
                if seq.status == Status.ABORTED:
                    continue
                seq.status = Status.RUNNING
                self.running.append(seq)
                seq.num_prefilled = n_tok                # 整段一次 prefill,准入即视作 prefill 完成
                out.prefill.append(seq)
                admitted += 1
                used_tokens += n_tok

        # 4) 其余 prefill 已完成的 running 序列前进一个 decode token
        #    (排除本步在做 prefill 的:整段的 out.prefill、分块的 out.prefill_chunk)。
        out.decode = [
            s for s in self.running
            if s is not out.prefill_chunk and s not in out.prefill
            and s.num_prefilled >= s.num_prompt_tokens
        ]
        return out

    def free_finished(self) -> List[Sequence]:
        """返回本步跑完的序列,好让引擎释放它们的 KV cache 并通知客户端。"""
        done = [s for s in self.running if s.status != Status.RUNNING]
        self.running = [s for s in self.running if s.status == Status.RUNNING]
        return done
