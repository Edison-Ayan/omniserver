"""为模型热点逐元素算子手写的融合 kernel(Triton)。

`rmsnorm` 提供融合的 RMSNorm kernel,以及 `patch_llm_rmsnorm`,把它替换进
语言模型栈、顶掉 transformers 的 Qwen2RMSNorm。
"""

from .rmsnorm import FusedRMSNorm, patch_llm_rmsnorm, rmsnorm

__all__ = ["FusedRMSNorm", "patch_llm_rmsnorm", "rmsnorm"]
