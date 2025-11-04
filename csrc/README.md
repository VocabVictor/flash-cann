# Flash-CANN C++ / CUDA / Ascend C Source Code

This directory contains backend implementations for Flash-Attention kernels across different hardware platforms.

## Directory Structure

```
csrc/
├── flash_attn/          # CUDA/NVIDIA GPU kernels (original flash-attention)
│   ├── src/            # Flash Attention CUDA kernels
│   └── ...
├── flash_attn_ck/       # AMD ROCm kernels using Composable Kernel
├── composable_kernel/   # AMD Composable Kernel library
├── cutlass/             # NVIDIA CUTLASS library for CUDA
├── fused_dense_lib/     # Fused dense layer implementations
├── layer_norm/          # Layer normalization kernels
├── npu/                 # **CANN/Ascend NPU kernels (to be implemented)**
│   ├── src/            # AscendC kernel implementations
│   ├── include/        # Header files
│   └── README.md       # NPU-specific documentation
└── README.md           # This file
```

## Backend Implementations

### CUDA Backend (NVIDIA GPUs)
- **Directory**: `flash_attn/`, `cutlass/`
- **Language**: CUDA C++
- **Hardware**: NVIDIA GPUs (Compute Capability >= 8.0 for optimal performance)
- **Status**: ✅ Complete (from original flash-attention)

### ROCm Backend (AMD GPUs)
- **Directory**: `flash_attn_ck/`, `composable_kernel/`
- **Language**: HIP (ROCm)
- **Hardware**: AMD GPUs
- **Status**: ✅ Complete (from original flash-attention)

### CANN Backend (Ascend NPUs)
- **Directory**: `npu/`
- **Language**: AscendC (Huawei's AI Core programming language)
- **Hardware**: Huawei Ascend NPUs (910, 910B, 310, etc.)
- **Status**: ⏳ To be implemented

## Development Guidelines

### For CANN/NPU Implementation:

When implementing CANN kernels in `npu/`, follow these patterns:

1. **Use AscendC Programming Model**:
   - "Copy In, Compute, Copy Out" pipeline
   - Utilize Cube Unit for matrix operations
   - Leverage Vector Unit for element-wise operations

2. **Memory Management**:
   - Global Memory (GM): 32GB+
   - L1 Buffer: 400KB-800KB per core
   - Unified Buffer (UB): 8-16MB per core

3. **Key APIs to Use**:
   - `matmul::Matmul<>` for batch matrix multiply
   - `SoftmaxFlashV2<>` for fused softmax
   - `DataCopy()`, `DataCopyPad()` for memory transfers
   - Event-based synchronization for multi-core coordination

4. **Reference Implementation**:
   - Study CANN samples: `/tmp/cann-samples/operator_contrib/FlashAttentionScoreSample/`
   - Existing CUDA implementation in `flash_attn/` directory

### Building

Each backend has its own build configuration:
- CUDA: Uses standard PyTorch extension build
- ROCm: Uses HIP compiler
- CANN: Requires CANN toolkit and AscendC compiler (to be configured)

## Integration with Python

Python bindings are located in:
- `flash_attn_2_cuda.py` - CUDA backend wrapper
- `flash_attn_triton_amd.py` - AMD Triton backend
- `flash_attn_cann.py` - CANN backend wrapper (to be implemented)

The main interface `flash_attn/flash_attn_interface.py` automatically selects the appropriate backend based on available hardware.

## References

- **Original Flash-Attention**: https://github.com/Dao-AILab/flash-attention
- **CANN Documentation**: https://www.hiascend.com/en/software/cann
- **CANN Samples**: https://gitee.com/ascend/samples
- **AscendC Programming Guide**: CANN operator development documentation
