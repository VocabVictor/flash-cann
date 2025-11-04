# Flash-Attention CANN/NPU Implementation

This directory contains the CANN (Compute Architecture for Neural Networks) implementation of Flash-Attention for Huawei Ascend NPUs.

## Status

⏳ **To be implemented** - Awaiting NPU hardware for development and testing

## Directory Structure (Planned)

```
npu/
├── src/
│   ├── flash_attn_fwd.cpp        # Forward kernel implementation
│   ├── flash_attn_bwd.cpp        # Backward kernel implementation
│   ├── flash_attn_varlen.cpp     # Variable-length sequence support
│   └── flash_attn_kvcache.cpp    # KV-cache implementation
├── include/
│   ├── flash_attn_npu.h          # Main header
│   └── utils.h                   # Utility functions
├── tiling/
│   └── flash_attn_tiling.h       # Tiling configuration
└── README.md                     # This file
```

## Implementation Strategy

### Based on CANN Research Findings:

1. **Leverage Existing CANN Flash Attention**:
   - CANN already provides a production Flash Attention implementation
   - Location: `/tmp/cann-samples/operator_contrib/FlashAttentionScoreSample/`
   - Strategy: Wrap existing CANN operators via PyTorch custom ops

2. **Architecture Pattern**:
   ```
   BMM1: Query × Key^T → Attention Scores (with scaling)
   Softmax: Scores → Weights
   BMM2: Weights × Value → Output
   ```

3. **Key CANN Components to Use**:
   - `matmul::Matmul<>` for batch matrix multiplication
   - `SoftmaxFlashV2<>` for fused softmax computation
   - Event-based synchronization for pipeline stages
   - Multi-core tiling for large sequences

## Hardware Architecture

### Ascend NPU Components:
- **AI Core**: Main compute unit
  - **Cube Unit**: 16×16 matrix multiplication (FP16/BF16)
  - **Vector Unit**: Element-wise operations, softmax
  - **Scalar Unit**: Control flow, indexing
- **Unified Buffer (UB)**: 8-16MB per core
- **L1 Buffer**: 400KB-800KB shared memory

### Memory Hierarchy:
```
Global Memory (GM) - 32GB+
    ↓
L1 Buffer - 400KB (shared)
    ↓
Unified Buffer (UB) - 8-16MB (per core)
    ↓
Cube/Vector Units
```

## Implementation Checklist

### Phase 1: Research & Setup (✅ Complete)
- [x] Study CANN operator development basics
- [x] Analyze existing Flash Attention in CANN samples
- [x] Understand AscendC programming model
- [x] Document hardware architecture

### Phase 2: PyTorch Integration (⏳ Pending NPU Hardware)
- [ ] Set up CANN development environment
- [ ] Install torch_npu library
- [ ] Create PyTorch custom operator bindings
- [ ] Implement tensor format conversion

### Phase 3: Kernel Implementation (⏳ Pending)
- [ ] Implement forward kernel wrapper
- [ ] Implement backward kernel wrapper
- [ ] Add variable-length sequence support
- [ ] Implement KV-cache functionality
- [ ] Optimize memory tiling

### Phase 4: Testing & Optimization (⏳ Pending)
- [ ] Unit tests for correctness
- [ ] Numerical accuracy validation
- [ ] Performance benchmarks vs CUDA
- [ ] Memory usage profiling

## Reference Implementation

The CANN sample Flash Attention implementation demonstrates:

```cpp
class FlashAttentionScoreS1s2Bn2gs1 {
    // Dual BMM objects
    matmul::Matmul<...> bmm1;  // Q×K^T
    matmul::Matmul<...> bmm2;  // Attn×V

    void Process() {
        // 3-stage pipeline
        IterateBmm1(extraInfo, bmm1);      // Compute Q×K^T
        ProcessVec1(extraInfo);            // Scale + Softmax
        IterateBmm2(extraInfo);            // Compute Attn×V
        ProcessVec2(extraInfo);            // Output processing
    }
};
```

### Key Features:
- **Tiling**: S1 (query seq), S2 (key seq), D (head dim)
- **Multi-core**: Task distribution across AI Cores
- **Double buffering**: Overlap compute and memory transfer
- **Event synchronization**: Cube ↔ Vector ↔ Memory coordination

## Data Types

- **Input**: FP16 (half precision)
- **Computation**: FP32 (single precision)
- **Output**: FP16 (half precision)

## Integration with flash_attn_cann.py

The Python wrapper will:
1. Check for `torch_npu` availability
2. Convert PyTorch tensors to NPU format
3. Call CANN operators via custom ops
4. Convert output back to PyTorch format

## Development Environment Requirements

- **OS**: Linux (Ubuntu 18.04+, EulerOS, CentOS)
- **CANN Toolkit**: Latest version
- **Python**: 3.7-3.8
- **torch_npu**: Matching PyTorch version
- **Hardware**: Ascend 910/910B/310 NPU

## References

- **CANN Samples**: https://gitee.com/ascend/samples
- **Flash Attention Paper**: https://arxiv.org/abs/2205.14135
- **CANN Documentation**: https://www.hiascend.com/en/software/cann
- **Original CUDA Implementation**: ../flash_attn/src/

## Notes

This implementation is on hold until NPU hardware becomes available for development and testing. The research phase has identified that:

1. **华为已有完整实现** - Huawei has complete Flash Attention in CANN samples
2. **应该封装而非重写** - Should wrap existing implementation rather than rewrite
3. **需要torch_npu集成** - Requires torch_npu for PyTorch integration
