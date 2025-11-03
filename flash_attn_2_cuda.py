"""
Fake CUDA backend for testing import logic.
"""

def fwd(*args, **kwargs):
    raise NotImplementedError("CUDA backend not installed")

def bwd(*args, **kwargs):
    raise NotImplementedError("CUDA backend not installed")

def varlen_fwd(*args, **kwargs):
    raise NotImplementedError("CUDA backend not installed")

def varlen_bwd(*args, **kwargs):
    raise NotImplementedError("CUDA backend not installed")

def fwd_kvcache(*args, **kwargs):
    raise NotImplementedError("CUDA backend not installed")
