# Flash-CANN

Flash-Attention implementation for Huawei Ascend NPU using CANN (Compute Architecture for Neural Networks).

将 Flash-Attention 算法移植到华为昇腾 NPU 平台，使用 CANN 异构计算架构实现。

## 项目简介 (Project Overview)

Flash-CANN 是 [Flash-Attention](https://github.com/Dao-AILab/flash-attention) 在华为昇腾 NPU 上的实现。Flash-Attention 是一种快速且内存高效的精确注意力算法，通过 IO 感知优化和分块计算技术，实现了：

- ⚡ **2-4倍速度提升**：相比标准 Attention
- 💾 **O(N) 空间复杂度**：从 O(N²) 降低到 O(N)
- 🚀 **减少 HBM 访问**：减少约 9倍的内存访问次数

## 核心技术 (Key Technologies)

### Flash-Attention 优化原理

1. **Tiling (分块计算)**：将大矩阵分块处理，避免实例化完整的 N×N 注意力矩阵
2. **Kernel Fusion (算子融合)**：将多个操作融合到一个 kernel，减少内存读写
3. **IO-Awareness (IO 感知)**：优化 HBM ↔ SRAM 之间的数据传输
4. **在线 Softmax**：使用统计量增量更新 softmax，无需存储中间结果

### GPU vs NPU 架构对比

| 特性 | NVIDIA GPU (CUDA) | Huawei Ascend NPU (CANN) |
|------|-------------------|--------------------------|
| 计算单元 | CUDA Cores + Tensor Cores | AI Core (Cube + Vector + Scalar) |
| 矩阵计算 | 多 Core 协作 / Tensor Core | Cube Unit (16×16 硬件矩阵乘) |
| 内存层次 | HBM ↔ Shared Memory | DDR ↔ L1 ↔ Unified Buffer |
| 编程模型 | CUDA (线程/块/网格) | Ascend C (AI Core 编程) |
| 设计理念 | 通用并行计算 (GPGPU) | AI 专用加速 |

## 项目目标 (Goals)

- [ ] 实现 Flash-Attention 前向传播 (Forward Pass)
- [ ] 实现 Flash-Attention 反向传播 (Backward Pass)
- [ ] 支持 FP16/BF16 数据类型
- [ ] 支持 Causal Masking
- [ ] 性能优化与基准测试
- [ ] Python 接口封装
- [ ] 与 PyTorch 集成

## 目录结构 (Project Structure)

```
flash-cann/
├── README.md           # 项目说明
├── csrc/              # C++/Ascend C 源代码
│   ├── kernels/       # CANN kernel 实现
│   └── operators/     # 算子封装
├── python/            # Python 接口
├── tests/             # 测试用例
├── benchmarks/        # 性能基准测试
└── docs/              # 文档
```

## 环境要求 (Requirements)

- 华为昇腾 NPU (Ascend 910/910B 推荐)
- CANN 工具链 >= 8.0
- Python >= 3.8
- (可选) MindStudio IDE

## 构建与安装 (Build & Installation)

```bash
# 待实现
# TBD
```

## 使用示例 (Usage)

```python
# 待实现
# TBD
```

## 技术挑战 (Technical Challenges)

### CUDA → CANN 移植要点

1. **并行模型转换**
   - GPU: 大量线程 (10k+) 处理小任务
   - NPU: 少量 AI Core，每个处理大块矩阵运算

2. **内存管理**
   - GPU Shared Memory → NPU Unified Buffer
   - 理解昇腾的三级存储层次

3. **算子映射**
   - CUDA Tensor Core → NPU Cube Unit (16×16 矩阵乘)
   - CUDA 向量运算 → NPU Vector Unit
   - 线程同步 → AI Core 调度

4. **性能优化**
   - 充分利用 Cube Unit 进行分块矩阵乘
   - 优化 Unified Buffer 使用
   - 减少 Global Memory 访问

## 参考资料 (References)

- [Flash-Attention 论文](https://arxiv.org/abs/2205.14135)
- [Flash-Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [CANN 官方文档](https://www.hiascend.com/document)
- [昇腾社区](https://www.hiascend.com/zh/)

## 开发状态 (Development Status)

🚧 **项目初期** - 正在规划架构和实现核心 kernel

## 贡献 (Contributing)

欢迎贡献代码、报告问题或提出建议！

## 许可证 (License)

待定 (TBD)

## 致谢 (Acknowledgments)

本项目基于 [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 的研究成果。
