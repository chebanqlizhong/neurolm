# AGENTS.md

面向 AI 代理与协作者的仓库专属说明。当构建命令、目录结构或约定发生变化时，
请保持本文件同步更新。

## 这是什么项目

NeuroLM：一个用纯 NumPy 从零实现、依赖极简的语言模型训练系统。不使用
PyTorch/TensorFlow。由一个初始的 hello-world 仓库改造而来。

## 环境

- Python 3.10+（开发环境为 3.13）。
- 依赖：`numpy`、`pyyaml`（运行时）；`pytest`、`ruff`（开发）。
- 无需 GPU，全部可在 CPU 上运行。

## 命令

```bash
pip install -e ".[dev]"        # 安装
python3 -m pytest -q           # 运行测试套件（36 个测试）
python3 -m ruff check src tests # 代码检查（CI 会执行）
make train-tiny                # 几秒，冒烟测试
make train-small               # CPU 约 90 秒
PYTHONPATH=src python3 -m neurolm <train|evaluate|generate|info> --config <cfg>
```

除非通过 `pip install -e .` 安装了包，否则需要设置 `PYTHONPATH=src`。

## 目录结构

- `src/neurolm/autograd.py` —— 反向模式自动求导引擎（系统基础）。
- `src/neurolm/{config,tokenizer,cli}.py` —— 配置、字节分词器、命令行。
- `src/neurolm/data/` —— token 数据集与批处理。
- `src/neurolm/models/` —— `Parameter`、`GPT` Transformer。
- `src/neurolm/training/` —— 优化器、训练器、检查点、评估。
- `configs/` —— YAML 运行配置；`data/` —— 语料；`tests/` —— pytest 测试套件。

## 约定

- 使用带类型标注的 Python，文件顶部写 `from __future__ import annotations`。
- 用 ruff 格式化/检查；CI 在 lint 报错时失败。
- 新增自动求导算子时，必须在 `tests/test_autograd.py` 中附带有限差分梯度校验。
- 配置是带类型的 dataclass；未知键必须报错。
- 文档与注释使用中文；代码、命令、文件名、标识符保持英文原样。
- 优先选择可读的代码而非炫技的代码；本项目的目的就是可学习。

## 已知坑

- `from __future__ import annotations` 会把 dataclass 字段类型变成字符串。
  `config._build` 使用 `typing.get_type_hints` 解析嵌套 dataclass —— 不要改回
  使用 `field.type`。
- 配置中 `batch_size` 位于 `data` 下，而不是 `train`。
- 训练产物（`runs/`、`*.npz`）已被 gitignore 忽略。

## 变更流程

1. 实现 + 添加/调整测试。
2. 必须通过 `python3 -m pytest -q` 与 `python3 -m ruff check src tests`。
3. 更新 `DEVELOPMENT_LOG.md`；对用户可见的变更还要更新 `CHANGELOG.md` 与
   `docs/ROADMAP.md`。
