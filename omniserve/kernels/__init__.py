"""Hand-written fused kernels (Triton) for the model's hot pointwise ops.

`rmsnorm` provides a fused RMSNorm kernel and `patch_llm_rmsnorm` to swap it into
the language-model stack in place of transformers' Qwen2RMSNorm.
"""

from .rmsnorm import FusedRMSNorm, patch_llm_rmsnorm, rmsnorm

__all__ = ["FusedRMSNorm", "patch_llm_rmsnorm", "rmsnorm"]
