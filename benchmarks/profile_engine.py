"""Profile where omniserve spends time on a concurrent multimodal workload.

Breaks wall time into prefill vs decode (the two model phases) vs scheduler/
overhead, with CUDA syncs for accurate GPU timing. This tells us which phase to
optimize first instead of guessing.

    python profile_engine.py --requests 24
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare._workload import MAX_NEW_TOKENS, PROMPT, build, make_image  # noqa: E402

from omniserve import LLMEngine, Request, SamplingParams, SchedulerConfig  # noqa: E402
from omniserve.runners.qwen_vl import QwenVLRunner  # noqa: E402


class Timed:
    """Wrap runner.prefill/decode to accumulate CUDA-accurate timings."""
    def __init__(self, runner):
        self.runner = runner
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.prefill_calls = 0
        self.decode_calls = 0
        self.prefill_seqs = 0
        self.decode_seqs = 0
        self._p = runner.prefill
        self._d = runner.decode

    def install(self):
        def prefill(seqs):
            torch.cuda.synchronize(); t = time.perf_counter()
            self._p(seqs)
            torch.cuda.synchronize()
            self.prefill_s += time.perf_counter() - t
            self.prefill_calls += 1; self.prefill_seqs += len(seqs)
        def decode(seqs):
            torch.cuda.synchronize(); t = time.perf_counter()
            self._d(seqs)
            torch.cuda.synchronize()
            self.decode_s += time.perf_counter() - t
            self.decode_calls += 1; self.decode_seqs += len(seqs)
        self.runner.prefill = prefill
        self.runner.decode = decode
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=24)
    ap.add_argument("--max-running", type=int, default=32)
    args = ap.parse_args()

    runner = QwenVLRunner(load_in_4bit=False)  # fp16, matches the comparison
    prof = Timed(runner).install()
    engine = LLMEngine(runner, SchedulerConfig(max_running=args.max_running,
                                               max_prefill_per_step=1))

    reqs = build(args.requests, reuse_rate=0.0)
    for r in reqs:
        engine.add_request(Request(prompt=PROMPT, images=[make_image(r.image_seed)],
                                   sampling=SamplingParams(max_new_tokens=MAX_NEW_TOKENS, temperature=0.0)))

    # warmup not needed for a breakdown; time the whole run
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    steps = 0
    while engine.has_unfinished():
        engine.step()
        steps += 1
    wall = time.perf_counter() - t0

    toks = sum(s.num_output_tokens for s in engine.sequences())
    other = wall - prof.prefill_s - prof.decode_s

    print(f"\n=== omniserve profile: {args.requests} reqs, fp16, {steps} steps ===")
    print(f"total wall            {wall:8.2f}s   ({toks} tokens, {toks/wall:.1f} tok/s)")
    print(f"  prefill             {prof.prefill_s:8.2f}s   {prof.prefill_s/wall:5.1%}   "
          f"{prof.prefill_calls} calls / {prof.prefill_seqs} seqs   "
          f"{prof.prefill_s/max(1,prof.prefill_seqs)*1000:.0f} ms/seq")
    print(f"  decode              {prof.decode_s:8.2f}s   {prof.decode_s/wall:5.1%}   "
          f"{prof.decode_calls} calls / {prof.decode_seqs} seq-steps   "
          f"{prof.decode_s/max(1,prof.decode_calls)*1000:.0f} ms/step "
          f"(avg batch {prof.decode_seqs/max(1,prof.decode_calls):.1f})")
    print(f"  scheduler/overhead  {other:8.2f}s   {other/wall:5.1%}")


if __name__ == "__main__":
    main()
