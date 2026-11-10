# 变更记录

本文件记录本项目的所有重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循[语义化版本](https://semver.org/lang/zh-CN/)。

## [未发布]

### 计划
- 子词（BPE）分词器。
- 从优化器状态断点续训，而不仅仅是权重。
- top-p（核）采样与重复惩罚。
- 机器可读的逐步指标（JSONL/CSV）。

## [0.1.0] - 2026-11-10

NeuroLM 的首个版本：一个从零实现、依赖极简的语言模型训练系统。

### 新增
- **自动求导引擎**（`neurolm.autograd`）：基于 NumPy 张量 `Tensor` 的反向模式
  自动微分，支持算术、矩阵乘、reshape、转置、归约、索引、`exp`/`log`，并为
  `softmax`、`log_softmax`、`layer_norm`、`gelu`、`embedding` 和 `cross_entropy`
  提供了复合节点。
- **配置**（`neurolm.config`）：带类型校验的 YAML 配置。
- **分词器**（`neurolm.tokenizer`）：字节级分词器，UTF-8 精确往返，含
  `<bos>`/`<eos>` 标记。
- **数据管道**（`neurolm.data`）：token 流数据集，训练/验证切分与随机连续块采样。
- **模型**（`neurolm.models`）：decoder-only Transformer，含 pre-norm 残差块、
  因果多头注意力、GELU 前馈网络以及可选的权重共享。
- **训练**（`neurolm.training`）：AdamW 优化器（解耦权重衰减）、带预热的余弦
  学习率调度、全局范数梯度裁剪、带验证与检查点的训练循环、损失/困惑度评估，
  以及温度/top-k 文本生成。
- **命令行**（`neurolm.cli`）：`train`、`evaluate`、`generate`、`info` 子命令。
- **配置**：`configs/tiny.yaml` 和 `configs/small.yaml`。
- **语料**：`data/corpus.txt`。
- **工程**：`Makefile`、`scripts/plot_history.py`、`scripts/train_and_sample.sh`、
  GitHub Actions CI。
- **文档**：`README.md`、`docs/ARCHITECTURE.md`、`docs/GETTING_STARTED.md`、
  `docs/ROADMAP.md`、`DEVELOPMENT_LOG.md`。
- **测试**：36 个测试，覆盖自动求导（含有限差分梯度校验）、数据、模型、配置与训练。

### 变更
- 仓库从 hello-world 演示改造为语言模型训练系统。`hello.py` 保留作为最小的
  工具链冒烟测试。
- 全部文档中文化，并使用 Mermaid 图表（架构图、数据流图、时序图、路线图）
  提升可读性。
- 命令行帮助、日志、用户可见的报错信息、Makefile/CI 注释全部中文化。
- 新增中文支持：`data/corpus_zh.txt`（中文语料）、`configs/zh.yaml`
  （中文配置）与 `make train-zh` 目标。

[未发布]: https://github.com/chebanqlizhong/neurolm/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/chebanqlizhong/neurolm/releases/tag/v0.1.0
