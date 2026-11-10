# 开发日志

按时间记录本项目改了什么、为什么改、验证了什么。最新的条目放在最上面。
每条都标注日期并链接到相关文件。

项目最初是一个 "hello world" 演示，后来被改造成一个从零实现的语言模型训练系统。
两个阶段都记录在下面。

---

## 2026-11-10 — 文档中文化 + 图表化

**目标.** 把仓库和文档改为中文，并通过图表（Mermaid 流程图、时序图、甘特式
路线图、数据流图）提升可读性与理解效率。

### 变更

| 文件 | 变更 |
|------|------|
| `README.md` | 全文中文；新增"全景图"数据流图、困惑度下降图；补充中文训练结果 |
| `docs/ARCHITECTURE.md` | 全文中文；新增分层架构图、前向/反向时序图、Transformer 块结构图、注意力形状流图、训练循环图、学习率调度图、配置加载图 |
| `docs/GETTING_STARTED.md` | 全文中文；新增上手流程图、产物结构图、排错决策树 |
| `docs/ROADMAP.md` | 全文中文；新增阶段进度图 |
| `DEVELOPMENT_LOG.md` | 全文中文 |
| `CHANGELOG.md` | 全文中文 |
| `AGENTS.md` | 全文中文 |
| `src/neurolm/cli.py` | 命令行帮助、日志与模块文档字符串中文化 |
| `src/neurolm/{config,data/dataset,models/gpt}.py` | 用户可见的报错信息中文化 |
| `scripts/plot_history.py`、`scripts/train_and_sample.sh` | 输出与注释中文化 |
| `Makefile`、`.github/workflows/ci.yml` | 注释与步骤名中文化；新增 `train-zh` 目标 |
| `configs/tiny.yaml`、`configs/small.yaml` | 配置注释中文化 |

### 新增：中文支持

- `data/corpus_zh.txt` —— 中文语料（约 4758 个 token），内容关于语言模型原理。
- `configs/zh.yaml` —— 中文训练配置（`block_size`/`max_seq_len` 更大，以适配
  汉字按字节展开后的更长序列）。
- `Makefile` 新增 `train-zh` 目标。

中文训练结果（200 步，约 100 秒）：

| 步数 | 训练损失 | 验证损失 | 验证困惑度 |
|-----:|---------:|---------:|-----------:|
| 0    | 5.551    | 5.518    | 249.2      |
| 50   | 2.880    | 2.983    | 19.7       |
| 100  | 2.548    | 2.773    | 16.0       |
| 199  | 2.153    | 2.571    | 13.1       |

### 说明

- 所有图表使用 GitHub 原生支持的 **Mermaid** 语法，无需额外工具即可在仓库
  页面渲染。
- 保留了代码、命令、文件名与标识符的英文原样，便于直接复制执行。
- 中文生成目前仍以"字"为单位、连贯性有限：字节级分词下每个汉字约占 3 个
  token，小模型 + 小语料难以学到词级搭配。改进方向见
  [路线图](docs/ROADMAP.md)（子词分词器 + 更大语料）。

---

## 2026-11-10 — NeuroLM v0.1.0：从零实现的语言模型训练系统

**目标.** 把仓库改造成一个专属 AI 模型的训练系统，具备合理的文件部署、
说明文档和可长期维护的开发日志。

### 构建内容

| 模块 | 文件 | 说明 |
|------|------|------|
| 自动求导 | `src/neurolm/autograd.py` | 反向模式引擎：算术、矩阵乘、reshape、转置、求和/均值、索引、exp/log、softmax、log-softmax、层归一化、GELU、embedding、交叉熵 |
| 配置 | `src/neurolm/config.py` | 类型化 dataclass（`ModelConfig`、`DataConfig`、`TrainConfig`、`RunConfig`），从 YAML 加载并校验 |
| 分词 | `src/neurolm/tokenizer.py` | 字节级分词器，`0..255` 加 `<bos>`/`<eos>` |
| 数据 | `src/neurolm/data/dataset.py` | 单数组 token 流、训练/验证切分、随机连续块采样 |
| 模型 | `src/neurolm/models/gpt.py`、`parameter.py` | Pre-norm decoder-only Transformer、因果多头注意力、GELU 前馈、权重共享 |
| 训练 | `src/neurolm/training/optimizer.py`、`trainer.py`、`checkpoint.py`、`evaluate.py` | AdamW、带预热的余弦学习率、梯度裁剪、训练循环、`.npz` 检查点、损失/困惑度、采样 |
| 接口 | `src/neurolm/cli.py`、`__main__.py` | `train`、`evaluate`、`generate`、`info` 子命令 |
| 配置 | `configs/tiny.yaml`、`configs/small.yaml` | tiny（几秒）与 small（CPU 约 90 秒） |
| 语料 | `data/corpus.txt` | 约 2.1 KB 关于语言模型的说明性文本 |
| 文档 | `README.md`、`docs/*` | 总览、架构、上手指南、路线图 |
| 工程 | `Makefile`、`scripts/*`、`.github/workflows/ci.yml` | 便捷目标、绘图、训练并采样、CI |
| 测试 | `tests/*` | 36 个测试，覆盖自动求导、数据、模型、配置、训练 |

### 验证

- **梯度正确性.** 对一个复合函数（层归一化 → GELU → 矩阵乘 → softmax）做有限
  差分对照，与解析梯度误差 < 2e-4
  （`tests/test_autograd.py::test_gradient_check_composite`）。
- **因果性.** 修改未来的 token 不会改变此前位置的 logits
  （`tests/test_model.py::test_causality_no_future_leakage`）。
- **学习能力.** 两个配置在自带语料上都能降低损失与困惑度。

`small` 运行（300 步，CPU 约 90 秒，394,722 个参数）：

| 步数 | 训练损失 | 验证损失 | 验证困惑度 |
|-----:|---------:|---------:|-----------:|
| 0    | 5.557    | 5.524    | 250.7      |
| 50   | 2.704    | 2.722    | 15.2       |
| 100  | 2.449    | 2.532    | 12.6       |
| 299  | 1.970    | 2.699    | 14.9       |

- **测试套件.** `pytest` → 36 项全部通过。

### 开发中发现并修复的问题

1. **嵌套配置未构建.** `from __future__ import annotations` 会把 dataclass 的
   字段类型变成字符串，导致 `is_dataclass(field.type)` 永远为 `False`，嵌套配置
   保持为 dict。改用 `typing.get_type_hints` 解析类型（见 `config._build`）。
2. **`batch_size` 读错配置.** 训练器去 `train.batch_size` 找，但它其实位于
   `data` 下。改为 `self.config.data.batch_size`。
3. **两个测试写错.** 一个交叉熵测试用了 2 维 logits，而实际需要 3 维
   `(batch, seq, vocab)`；一个 AdamW 测试断言了错误的不变量。均已修正。

### 说明 / 决策

- **选用字节级分词器**：零配置、无未登录词、精确往返；子词分词已列入
  [路线图](docs/ROADMAP.md)。
- **不依赖任何框架.** 项目的目的就是透明，只用 NumPy。
- **`small` 配置面向 CPU 调优**，使完整训练约 90 秒完成。此前更重的版本
  （d_model 128、4 层、600 步）会超出 5 分钟预算。

---

## 2026-11-10 — 项目起步（hello world）

**目标.** 创建一个 GitHub 仓库和一个最小的 hello 程序。

- 创建了 `hello-world` 仓库。
- 添加了 `hello.py`（输出 `Hello, World!`）和一个简短的 `README.md`。
- 初始提交推送到 `main`。

这一阶段作为仓库的第一次提交保留下来；代码已被上面的训练系统取代。
`hello.py` 保留在仓库根目录，作为一个验证工具链可用的最小冒烟测试。
