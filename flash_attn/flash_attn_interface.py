"""
Flash-Attention interface for multiple backends.

Supports:
- CUDA backend (NVIDIA GPUs) - via flash_attn_2_cuda
- ROCm/Triton backend (AMD GPUs) - via flash_attn_triton_amd
- CANN backend (Huawei Ascend NPUs) - via flash_attn_cann

Backend selection priority:
1. Check device type (cuda/npu)
2. Check environment variables
3. Fallback to reference implementation
"""

import math
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# isort: off
# We need to import the backends after importing torch

# Backend detection
def _get_backend():
    """
    Determine which backend to use based on device and environment.

    Returns:
        backend_module: The backend module to use
        backend_name: Name of the backend ('cuda', 'triton_amd', 'cann', 'reference')
    """
    # Force specific backend via environment variable
    force_backend = os.getenv("FLASH_ATTENTION_BACKEND", None)

    if force_backend == "REFERENCE":
        return None, "reference"

    # Try CANN backend for Ascend NPU
    if force_backend == "CANN" or torch.cuda.is_available():
        # Check if we're on Ascend NPU
        try:
            import torch_npu
            # If torch_npu is available, try to use CANN backend
            try:
                import flash_attn_cann
                print("[Flash-Attention] Using CANN backend for Ascend NPU")
                return flash_attn_cann, "cann"
            except ImportError:
                print("[Flash-Attention] CANN backend not available, falling back to reference")
                return None, "reference"
        except ImportError:
            pass

    # Try AMD ROCm/Triton backend
    use_triton_amd = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
    if use_triton_amd or force_backend == "TRITON_AMD":
        try:
            from flash_attn.flash_attn_triton_amd import interface_fa as flash_attn_triton_amd
            print("[Flash-Attention] Using Triton/AMD backend")
            return flash_attn_triton_amd, "triton_amd"
        except ImportError:
            print("[Flash-Attention] Triton/AMD backend not available")

    # Try CUDA backend (default for NVIDIA GPUs)
    if force_backend == "CUDA" or force_backend is None:
        try:
            import flash_attn_2_cuda
            print("[Flash-Attention] Using CUDA backend for NVIDIA GPU")
            return flash_attn_2_cuda, "cuda"
        except ImportError:
            print("[Flash-Attention] CUDA backend not available")

    # Fallback to reference implementation
    print("[Flash-Attention] Using reference PyTorch implementation (slow)")
    return None, "reference"


flash_attn_backend, backend_name = _get_backend()

# isort: on


def _flash_attn_forward_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation of flash attention in pure PyTorch.

    This is SLOW but correct, used for:
    - Testing correctness of backend implementations
    - Fallback when no backend is available
    - Understanding the algorithm

    Args:
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        v: (batch, seqlen, nheads, headdim)
        softmax_scale: scaling factor for QK^T
        causal: whether to apply causal mask

    Returns:
        out: (batch, seqlen, nheads, headdim)
        softmax_lse: (batch, nheads, seqlen) - log-sum-exp
    """
    # Transpose to (batch, nheads, seqlen, headdim) for bmm
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Compute attention scores: QK^T
    # (batch, nheads, seqlen, seqlen)
    scores = torch.matmul(q, k.transpose(-2, -1)) * softmax_scale

    # Apply causal mask if needed
    if causal:
        seqlen = q.size(2)
        causal_mask = torch.triu(
            torch.ones(seqlen, seqlen, dtype=torch.bool, device=q.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

    # Softmax
    attn = F.softmax(scores, dim=-1)

    # Compute log-sum-exp for backward pass
    # softmax_lse = log(sum(exp(scores)))
    max_scores = scores.max(dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - max_scores)
    sum_exp = exp_scores.sum(dim=-1)
    softmax_lse = torch.log(sum_exp) + max_scores.squeeze(-1)

    # Attention output
    out = torch.matmul(attn, v)

    # Transpose back to (batch, seqlen, nheads, headdim)
    out = out.transpose(1, 2).contiguous()

    return out, softmax_lse


def maybe_contiguous(x):
    """Ensure tensor is contiguous if needed."""
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """
    Flash Attention forward pass with multi-backend support.

    Supports CUDA (NVIDIA), ROCm/Triton (AMD), CANN (Ascend NPU), and reference backends.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim) - Query tensor
        k: (batch_size, seqlen, nheads_k, headdim) - Key tensor
        v: (batch_size, seqlen, nheads_k, headdim) - Value tensor
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size for local attention.
            (-1, -1) means no windowing (global attention).
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (batch_size, seqlen, nheads, headdim)
        softmax_lse: (batch_size, nheads, seqlen) - Log-sum-exp (if return_attn_probs)
        S_dmask: (batch_size, nheads, seqlen, seqlen) - Attention probs (if return_attn_probs)
    """
    # Validate inputs
    assert q.dim() == 4, f"q must be 4D, got {q.dim()}D"
    assert k.dim() == 4, f"k must be 4D, got {k.dim()}D"
    assert v.dim() == 4, f"v must be 4D, got {v.dim()}D"

    batch_size, seqlen_q, nheads, headdim = q.shape
    _, seqlen_k, nheads_k, _ = k.shape

    # Set default softmax scale
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(headdim)

    # Use backend-specific implementation
    if backend_name == "reference":
        # Reference implementation (limited features)
        if dropout_p > 0.0:
            raise NotImplementedError("Dropout not supported in reference backend")
        if window_size != (-1, -1):
            raise NotImplementedError("Window not supported in reference backend")
        if alibi_slopes is not None:
            raise NotImplementedError("ALiBi not supported in reference backend")

        out, softmax_lse = _flash_attn_forward_reference(q, k, v, softmax_scale, causal)

        if return_attn_probs:
            return out, softmax_lse, None
        else:
            return out

    elif backend_name == "cann":
        # CANN backend for Ascend NPU
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        out, softmax_lse, S_dmask, rng_state = flash_attn_backend.fwd(
            q, k, v,
            None,  # bias
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0], window_size[1],
            0.0,  # softcap
            return_attn_probs,
        )

        if return_attn_probs:
            return out, softmax_lse, S_dmask
        else:
            return out

    elif backend_name in ["cuda", "triton_amd"]:
        # CUDA or AMD backend
        q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

        out, softmax_lse, S_dmask, rng_state = flash_attn_backend.fwd(
            q, k, v,
            None,  # bias
            alibi_slopes,
            dropout_p,
            softmax_scale,
            causal,
            window_size[0], window_size[1],
            0.0,  # softcap
            return_attn_probs,
        )

        if return_attn_probs:
            return out, softmax_lse, S_dmask
        else:
            return out

    else:
        raise RuntimeError(f"Unknown backend: {backend_name}")


def flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Flash Attention with packed KV tensors."""
    # Unpack KV
    k, v = kv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic, return_attn_probs
    )


def flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Flash Attention with packed QKV tensors."""
    # Unpack QKV
    q, k, v = qkv.unbind(dim=2)
    return flash_attn_func(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, alibi_slopes, deterministic, return_attn_probs
    )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Flash Attention with variable length sequences."""
    raise NotImplementedError("flash_attn_varlen_func not yet implemented for CANN backend")


def flash_attn_varlen_kvpacked_func(
    q,
    kv,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Flash Attention with variable length sequences and packed KV."""
    raise NotImplementedError("flash_attn_varlen_kvpacked_func not yet implemented for CANN backend")


def flash_attn_varlen_qkvpacked_func(
    qkv,
    cu_seqlens,
    max_seqlen,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
    """Flash Attention with variable length sequences and packed QKV."""
    raise NotImplementedError("flash_attn_varlen_qkvpacked_func not yet implemented for CANN backend")


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens=None,
    cache_batch_idx=None,
    cache_leftpad=None,
    block_table=None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
):
    """Flash Attention with KV cache for autoregressive generation."""
    raise NotImplementedError("flash_attn_with_kvcache not yet implemented for CANN backend")
