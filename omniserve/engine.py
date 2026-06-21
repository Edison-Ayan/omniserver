"""推理引擎:驱动「调度器 / runner」的步进循环。

一次 `step()` 就是一轮 continuous batching:
    调度 -> prefill 新序列 + decode 在跑的序列 -> 退休跑完的。

每一步返回各序列新产生的增量,这样 server 可以一生成就把 token 流式发给客户端。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .runners.base import ModelRunner
from .request import Request, Sequence, Status
from .scheduler import Scheduler, SchedulerConfig


@dataclass
class StepDelta:
    """一步里某个序列产生的输出增量,用于流式回传客户端。"""
    request_id: str
    text: str
    finished: bool


class LLMEngine:
    def __init__(self, runner: ModelRunner, sched_config: SchedulerConfig | None = None):
        self.runner = runner
        self.scheduler = Scheduler(sched_config or SchedulerConfig())
        self._seqs: Dict[str, Sequence] = {}
        # 保留提交过的所有序列(跑完的会从 _seqs 里清掉);跑完后统计指标时方便。
        self._history: Dict[str, Sequence] = {}

    def add_request(self, request: Request) -> str:
        seq = Sequence(request=request)
        self.runner.tokenize(seq)
        self._seqs[seq.request_id] = seq
        self._history[seq.request_id] = seq
        self.scheduler.add(seq)
        return seq.request_id

    def sequences(self):
        """提交给本引擎的所有序列(在跑的 + 跑完的)。"""
        return list(self._history.values())

    def abort(self, request_id: str) -> None:
        self.scheduler.abort(request_id)

    def has_unfinished(self) -> bool:
        return self.scheduler.has_work()

    def step(self) -> List[StepDelta]:
        out = self.scheduler.schedule()
        deltas: List[StepDelta] = []

        if out.prefill:
            self.runner.prefill(out.prefill)
        if out.prefill_chunk is not None and out.decode:
            # Phase B:有 chunk 又有 decode -> 拼进同一次前向(mixed batching)
            self.runner.mixed_step(out.prefill_chunk, out.decode)
        elif out.prefill_chunk is not None:
            self.runner.chunk_prefill(out.prefill_chunk)   # 没有 decode 时纯算块
        elif out.decode:
            self.runner.decode(out.decode)                 # 纯 decode(可走 CUDA graph)

        # 收集本步动过的每个序列新流出的 token。
        touched = (*out.prefill,
                   *((out.prefill_chunk,) if out.prefill_chunk is not None else ()),
                   *out.decode)
        for seq in touched:
            new_ids = seq.output_token_ids[seq.num_streamed:]
            if new_ids:
                text = self.runner.detokenize(seq, new_ids)
                seq.num_streamed = seq.num_output_tokens
                deltas.append(StepDelta(seq.request_id, text, seq.status != Status.RUNNING))

        # 释放跑完的序列:让 runner 回收它们的 KV 槽位(它会紧凑化预分配池),
        # 再从注册表里删掉。
        for seq in self.scheduler.free_finished():
            self.runner.free(seq)
            self._seqs.pop(seq.request_id, None)

        return deltas

    def run_until_complete(self) -> Dict[str, str]:
        """离线/批量使用和测试用的同步辅助方法。"""
        texts: Dict[str, List[str]] = {rid: [] for rid in self._seqs}
        while self.has_unfinished():
            for d in self.step():
                texts.setdefault(d.request_id, []).append(d.text)
        return {rid: "".join(parts) for rid, parts in texts.items()}
