"""
Flash-CANN: Flash-Attention implementation for Huawei Ascend NPU

This is a drop-in replacement for flash-attn that uses CANN backend
instead of CUDA for Huawei Ascend NPU acceleration.

Based on: https://github.com/Dao-AILab/flash-attention
"""

__version__ = "0.1.0"

from flash_attn.flash_attn_interface import (
    flash_attn_func,
    flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_with_kvcache,
)

__all__ = [
    "__version__",
    "flash_attn_func",
    "flash_attn_kvpacked_func",
    "flash_attn_qkvpacked_func",
    "flash_attn_varlen_func",
    "flash_attn_varlen_kvpacked_func",
    "flash_attn_varlen_qkvpacked_func",
    "flash_attn_with_kvcache",
]
