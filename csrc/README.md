# Flash-CANN C++ / Ascend C Source Code

This directory contains the CANN implementation of Flash-Attention kernels.

## Structure

```
csrc/
├── flash_attn/          # Main flash attention kernels
│   ├── src/            # Kernel implementations
│   └── flash_api.cpp   # C++ API wrapper
└── README.md           # This file
```

## Development

CANN kernels will be written in Ascend C, Huawei's programming language
for AI Core development on Ascend NPU.

### Key Components to Implement:

- [ ] Forward kernel (flash_fwd_kernel)
- [ ] Backward kernel (flash_bwd_kernel)
- [ ] Softmax computation
- [ ] Block-wise tiling logic
- [ ] Memory management (Unified Buffer)
