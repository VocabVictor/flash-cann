"""
Fake Triton/AMD backend for testing import logic.
"""

class interface_fa:
    @staticmethod
    def fwd(*args, **kwargs):
        raise NotImplementedError("Triton/AMD backend not installed")

    @staticmethod
    def bwd(*args, **kwargs):
        raise NotImplementedError("Triton/AMD backend not installed")

    @staticmethod
    def varlen_fwd(*args, **kwargs):
        raise NotImplementedError("Triton/AMD backend not installed")

    @staticmethod
    def varlen_bwd(*args, **kwargs):
        raise NotImplementedError("Triton/AMD backend not installed")

    @staticmethod
    def fwd_kvcache(*args, **kwargs):
        raise NotImplementedError("Triton/AMD backend not installed")
