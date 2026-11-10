.PHONY: help install dev test lint format train-small train-tiny train-zh generate clean

PYTHON ?= python3
export PYTHONPATH := src

help: ## 显示可用的目标
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## 安装包（运行时依赖）
	$(PYTHON) -m pip install -e .

dev: ## 安装包（含开发依赖）
	$(PYTHON) -m pip install -e ".[dev]"

test: ## 运行测试套件
	$(PYTHON) -m pytest

lint: ## 检查源码
	$(PYTHON) -m ruff check src tests

format: ## 自动格式化源码
	$(PYTHON) -m ruff format src tests

train-tiny: ## 训练 tiny 配置（快速冒烟测试）
	$(PYTHON) -m neurolm train --config configs/tiny.yaml

train-small: ## 训练 small 配置
	$(PYTHON) -m neurolm train --config configs/small.yaml

train-zh: ## 训练中文配置
	$(PYTHON) -m neurolm train --config configs/zh.yaml

generate: ## 从 small 运行采样（可用 PROMPT=... 覆盖）
	$(PYTHON) -m neurolm generate --config configs/small.yaml \
		--checkpoint runs/small/final.npz --prompt "$(PROMPT)" --max-new-tokens 200

clean: ## 清理缓存与训练产物
	rm -rf runs .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
