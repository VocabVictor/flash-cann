"""
Flash-Attention interface functions for CANN backend.

This module provides the main API functions that are compatible with
the original flash-attn package.

All functions currently return NotImplementedError and will be
implemented progressively.
"""


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
    Flash Attention forward pass using CANN backend.

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
        softmax_lse: (batch_size, nheads, seqlen) - Log-sum-exp of softmax
        S_dmask: (batch_size, nheads, seqlen, seqlen) - Attention probs (if return_attn_probs=True)
    """
    raise NotImplementedError("flash_attn_func not yet implemented for CANN backend")


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
    """
    Flash Attention with packed KV tensors.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        kv: (batch_size, seqlen, 2, nheads_k, headdim) - Packed key and value
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (batch_size, seqlen, nheads, headdim)
    """
    raise NotImplementedError("flash_attn_kvpacked_func not yet implemented for CANN backend")


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
    """
    Flash Attention with packed QKV tensors.

    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, headdim) - Packed query, key, and value
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (batch_size, seqlen, nheads, headdim)
    """
    raise NotImplementedError("flash_attn_qkvpacked_func not yet implemented for CANN backend")


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
    """
    Flash Attention with variable length sequences.

    Arguments:
        q: (total_q, nheads, headdim) - Query tensor (sequences concatenated)
        k: (total_k, nheads_k, headdim) - Key tensor
        v: (total_k, nheads_k, headdim) - Value tensor
        cu_seqlens_q: (batch_size + 1,) - Cumulative sequence lengths for queries
        cu_seqlens_k: (batch_size + 1,) - Cumulative sequence lengths for keys
        max_seqlen_q: int. Maximum sequence length for queries
        max_seqlen_k: int. Maximum sequence length for keys
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (total_q, nheads, headdim)
    """
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
    """
    Flash Attention with variable length sequences and packed KV.

    Arguments:
        q: (total_q, nheads, headdim)
        kv: (total_k, 2, nheads_k, headdim) - Packed key and value
        cu_seqlens_q: (batch_size + 1,)
        cu_seqlens_k: (batch_size + 1,)
        max_seqlen_q: int
        max_seqlen_k: int
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (total_q, nheads, headdim)
    """
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
    """
    Flash Attention with variable length sequences and packed QKV.

    Arguments:
        qkv: (total, 3, nheads, headdim) - Packed query, key, and value
        cu_seqlens: (batch_size + 1,) - Cumulative sequence lengths
        max_seqlen: int. Maximum sequence length
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        deterministic: bool. Whether to use deterministic implementation.
        return_attn_probs: bool. Whether to return attention probabilities.

    Returns:
        out: (total, nheads, headdim)
    """
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
    """
    Flash Attention with KV cache for autoregressive generation.

    Arguments:
        q: (batch_size, seqlen_q, nheads, headdim) or (batch_size, 1, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_k, nheads_k, headdim) or
                 (num_blocks, page_block_size, nheads_k, headdim) for paged attention
        v_cache: (batch_size_cache, seqlen_k, nheads_k, headdim) or
                 (num_blocks, page_block_size, nheads_k, headdim)
        k: (batch_size, seqlen_knew, nheads_k, headdim) - New keys to append to cache
        v: (batch_size, seqlen_knew, nheads_k, headdim) - New values to append to cache
        rotary_cos: (seqlen_ro, rotary_dim / 2) - RoPE cos
        rotary_sin: (seqlen_ro, rotary_dim / 2) - RoPE sin
        cache_seqlens: (batch_size,) - Sequence lengths in cache
        cache_batch_idx: (batch_size,) - Batch indices for cache
        cache_leftpad: (batch_size,) - Left padding in cache
        block_table: (batch_size, max_num_blocks_per_seq) - For paged attention
        softmax_scale: float. The scaling of QK^T before applying softmax.
        causal: bool. Whether to apply causal attention mask.
        window_size: (int, int). Left and right window size.
        rotary_interleaved: bool. RoPE interleaved format.
        alibi_slopes: (nheads,) or (batch_size, nheads). ALiBi slopes.
        num_splits: int. Number of splits for parallelization.

    Returns:
        out: (batch_size, seqlen_q, nheads, headdim)
    """
    raise NotImplementedError("flash_attn_with_kvcache not yet implemented for CANN backend")
