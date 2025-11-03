"""
Fake CANN backend for testing import logic.
This will be replaced with real CANN kernel implementation.
"""

def fwd(*args, **kwargs):
    """Fake forward pass"""
    raise NotImplementedError("CANN backend fwd() not yet implemented")

def bwd(*args, **kwargs):
    """Fake backward pass"""
    raise NotImplementedError("CANN backend bwd() not yet implemented")

def varlen_fwd(*args, **kwargs):
    """Fake varlen forward pass"""
    raise NotImplementedError("CANN backend varlen_fwd() not yet implemented")

def varlen_bwd(*args, **kwargs):
    """Fake varlen backward pass"""
    raise NotImplementedError("CANN backend varlen_bwd() not yet implemented")

def fwd_kvcache(*args, **kwargs):
    """Fake kvcache forward pass"""
    raise NotImplementedError("CANN backend fwd_kvcache() not yet implemented")
