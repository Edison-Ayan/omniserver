"""GEMM micro-bench(真实模型版):真实 Qwen2-VL-2B 权重 + 真实激活的速度 / 精度损失。

bench_gemm.py 用随机权重+随机激活——对**速度**有效(GEMM 延迟只看形状),但**精度损失不可信**
(量化误差对真实分布、尤其激活 outlier 极敏感)。本脚本把两者都换成真实的:

  - 真实权重:transformers 加载本地 Qwen2-VL-2B-Instruct(fp16),按 omniserve 的融合顺序
              concat 成 qkv=[q,k,v] / gate_up=[gate,up] / o / down,对齐 bench_gemm.py 的形状。
  - 真实激活:forward_pre_hook 抓中间层(默认 layer 14)四个 proj 的真实输入。其中
              **down 的输入是 SwiGLU 中间激活(silu(gate)*up)**——W8A8 的 outlier 重灾区,
              随机高斯完全测不出来。

两张表:
  1. 速度(扫 M=24/512/1024):真实权重,激活数值不影响延迟,用于交叉验证 bench_gemm.py。
  2. 精度损失(真实 M=prompt 长度):真实权重×真实激活对 fp16 的偏差(rel L2 + 逐行 cos)。
     FP8 给两档粒度:per-tensor 和 rowwise(per-token 激活 + per-channel 权重)——看粒度
     能否压住真实激活 outlier。W4A16 是无校准 RTN(GPTQ 校准会更好,属后续)。

⚠️ 单 GEMM 的 rel err 是 proxy;端到端 token 接受率/perplexity 要全模型量化后另测(子任务后续)。

用法:
  conda activate vllm_bench
  HF_HUB_OFFLINE=1 python bench_gemm_real.py            # layer 14
  HF_HUB_OFFLINE=1 python bench_gemm_real.py --layer 0  # 看不同深度
"""

from __future__ import annotations

import argparse
import os
import sys
from statistics import median

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from omniserve.kernels.marlin_linear import MarlinInt4Linear  # noqa: E402

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
M_LIST = [24, 512, 1024]
E4M3_MAX = 448.0

# 一段够长的真实文本(中英混合,贴近真实使用),让 prefill 激活有代表性
PROMPT = (
    "请详细解释大语言模型推理中的 KV cache、PagedAttention 与连续批处理是如何协同降低显存碎片、"
    "提升吞吐的,并对比 prefill 与 decode 两个阶段在算力受限与带宽受限上的本质差异。"
    "In modern LLM serving, quantization such as GPTQ, AWQ and FP8 trades numerical precision for "
    "memory bandwidth and compute throughput; explain why weight-only int4 helps the bandwidth-bound "
    "decode stage while W8A8 FP8 shines in the compute-bound prefill stage, and how activation "
    "outliers (the motivation behind SmoothQuant) make per-tensor scaling fragile compared to "
    "per-channel or per-token granularity. 结合 roofline 模型,从 GEMM 的 M 维如何决定瓶颈这一角度,"
    "论证为什么最优精度配置不是全局单一精度,而是 per-op × 阶段 × 数值敏感度 的函数。"
) * 2


def cuda_time_us(fn, iters=100, warmup=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        samples.append(s.elapsed_time(e))
    return median(samples) * 1000.0


def to_fp8_per_tensor(t):
    amax = t.abs().max().clamp(min=1e-4)
    scale = (amax / E4M3_MAX).to(torch.float32)
    return (t / scale).to(torch.float8_e4m3fn), scale


def make_fp8_per_tensor(w):
    """W8A8 per-tensor。权重离线量化一次(静态,不计时);返回只在线量化激活的 forward。"""
    w8, sw = to_fp8_per_tensor(w)
    wb = w8.contiguous().t()                                        # [K,N] 列主序(不能再 contiguous)

    def run(x):
        a8, sa = to_fp8_per_tensor(x)                               # 激活在线量化(计时内)
        return torch._scaled_mm(a8, wb, scale_a=sa, scale_b=sw, out_dtype=torch.float16)
    return run


def make_fp8_rowwise(w):
    """W8A8 rowwise:权重 per-output-channel(静态)+ 激活 per-token(在线)。治 outlier。
    注:rowwise scaled_mm 只支持 bf16 输出。"""
    w_amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)       # [N,1]
    sw = (w_amax / E4M3_MAX).to(torch.float32)
    w8 = (w / sw).to(torch.float8_e4m3fn)
    wb = w8.contiguous().t()                                        # [K,N] 列主序
    swT = sw.t().contiguous()                                       # [1,N]

    def run(x):
        a_amax = x.abs().amax(dim=1, keepdim=True).clamp(min=1e-4)  # [M,1]
        sa = (a_amax / E4M3_MAX).to(torch.float32)
        a8 = (x / sa).to(torch.float8_e4m3fn)
        return torch._scaled_mm(a8, wb, scale_a=sa, scale_b=swT, out_dtype=torch.bfloat16)
    return run


def quality(out, ref):
    """返回 (rel L2 误差, 逐行 cos 相似度均值)。"""
    rel = (out.float() - ref.float()).norm() / ref.float().norm()
    cos = F.cosine_similarity(out.float(), ref.float(), dim=-1).mean()
    return rel.item(), cos.item()


def grab_real_weights_and_acts(layer_idx: int, use_image: bool = False, image_path: str = None):
    """加载真实模型,抓 layer_idx 的真实激活,返回融合权重 + 真实激活,然后释放模型。
    use_image=True:喂真实图文(走 ViT,激活含图像 token);image_path 给则用真实照片,否则合成几何图。"""
    from transformers import Qwen2VLForConditionalGeneration
    print(f"加载 {MODEL_ID}(fp16)... [{'图文' if use_image else '纯文本'}]")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, dtype=torch.float16, device_map="cuda").eval()
    layer = model.model.language_model.layers[layer_idx]

    acts = {}
    def pre_hook(name):
        def h(_mod, inp):
            acts[name] = inp[0].detach().reshape(-1, inp[0].shape[-1])  # [M,K]
        return h
    hs = [
        layer.self_attn.q_proj.register_forward_pre_hook(pre_hook("qkv")),   # qkv 输入
        layer.self_attn.o_proj.register_forward_pre_hook(pre_hook("o")),     # attn 输出
        layer.mlp.gate_proj.register_forward_pre_hook(pre_hook("gate_up")),  # gate_up 输入
        layer.mlp.down_proj.register_forward_pre_hook(pre_hook("down")),     # SwiGLU 中间激活
    ]
    with torch.no_grad():
        if use_image:  # 真实图文:processor 处理图片 + 文本,完整 forward(含 ViT)
            from transformers import AutoProcessor
            if image_path:  # 真实照片
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                img.thumbnail((1024, 1024))  # 限分辨率:超大图会让 ViT(O(N²))撑爆 8GB
                print(f"  真实照片:{image_path} -> {img.size}")
            else:           # 项目端到端 bench 同款合成几何图
                sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare"))
                from _workload import make_image
                img = make_image(0)
            proc = AutoProcessor.from_pretrained(MODEL_ID)
            msgs = [{"role": "user", "content": [{"type": "image"},
                     {"type": "text", "text": "Describe this image in detail."}]}]
            text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            model(**inputs)
        else:  # 纯文本
            from transformers import AutoTokenizer
            ids = AutoTokenizer.from_pretrained(MODEL_ID)(PROMPT, return_tensors="pt").input_ids.to("cuda")
            model(input_ids=ids)
    for h in hs:
        h.remove()
    print(f"真实激活 M = {acts['qkv'].shape[0]}")

    sa, mlp = layer.self_attn, layer.mlp
    # 按 omniserve 融合顺序拼真实权重(qkv=[q,k,v]、gate_up=[gate,up])
    W = {
        "qkv": (torch.cat([sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight], 0).detach(),
                torch.cat([sa.q_proj.bias, sa.k_proj.bias, sa.v_proj.bias], 0).detach()),
        "o": (sa.o_proj.weight.detach(), None),
        "gate_up": (torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).detach(), None),
        "down": (mlp.down_proj.weight.detach(), None),
    }
    W = {k: (w.contiguous().half(), (b.contiguous().half() if b is not None else None))
         for k, (w, b) in W.items()}
    X = {k: v.contiguous().half() for k, v in acts.items()}
    del model
    torch.cuda.empty_cache()
    return W, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=14, help="抓哪一层的真实激活(0..27)")
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  sm_{p.major}{p.minor}  torch {torch.__version__}")
    W, X = grab_real_weights_and_acts(args.layer)

    order = ["qkv", "o", "gate_up", "down"]
    # 预构造每个 GEMM 的真实 fp16 Linear + Marlin(RTN)+ FP8 runner(权重静态量化)
    lin, marlin, fp8t, fp8r = {}, {}, {}, {}
    for name in order:
        w, b = W[name]
        N, K = w.shape
        L = torch.nn.Linear(K, N, bias=b is not None).half().cuda().eval()
        L.weight.data.copy_(w)
        if b is not None:
            L.bias.data.copy_(b)
        lin[name] = L
        marlin[name] = MarlinInt4Linear(L)
        fp8t[name] = make_fp8_per_tensor(w)
        fp8r[name] = make_fp8_rowwise(w)

    # ---- 表1:速度(真实权重,扫 M;激活数值不影响延迟)----
    print("\n=== 表1 速度(真实权重,加速比相对 fp16)===")
    print(f"{'GEMM':8} {'K':>5} {'N':>6} {'M':>5} | {'fp16':>8} {'W4A16':>8} {'FP8t':>8} | "
          f"{'W4A16↑':>7} {'FP8↑':>6}")
    for name in order:
        w, _ = W[name]
        N, K = w.shape
        L = lin[name]
        for M in M_LIST:
            x = torch.randn(M, K, device="cuda", dtype=torch.float16)
            t16 = cuda_time_us(lambda: L(x), args.iters)
            t4 = cuda_time_us(lambda: marlin[name].forward(x), args.iters)
            t8 = cuda_time_us(lambda: fp8t[name](x), args.iters)
            print(f"{name:8} {K:>5} {N:>6} {M:>5} | {t16:8.1f} {t4:8.1f} {t8:8.1f} | "
                  f"{t16/t4:6.2f}x {t16/t8:5.2f}x")

    # ---- 表2:精度损失(真实权重 × 真实激活,真实 M)----
    print(f"\n=== 表2 精度损失(真实权重 × 真实激活,layer {args.layer},"
          f"M={X['qkv'].shape[0]})===")
    print(f"{'GEMM':8} | {'W4A16 rel/cos':>18} | {'FP8-tensor rel/cos':>20} | "
          f"{'FP8-row rel/cos':>18}")
    for name in order:
        w, b = W[name]
        x = X[name]
        ref = lin[name](x)
        r4, c4 = quality(marlin[name].forward(x), ref)
        out_t = fp8t[name](x) + (b if b is not None else 0)
        rt, ct = quality(out_t, ref)
        out_r = fp8r[name](x) + (b if b is not None else 0)
        rr, cr = quality(out_r, ref)
        print(f"{name:8} | {r4:7.4f} / {c4:6.4f} | {rt:8.4f} / {ct:6.4f}   | "
              f"{rr:7.4f} / {cr:6.4f}")


if __name__ == "__main__":
    main()
