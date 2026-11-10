.PHONY: help install dev test lint format train-small train-tiny generate clean

PYTHON ?= python3
export PYTHONPATH := src

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install the package (runtime dependencies)
	$(PYTHON) -m pip install -e .

dev: ## Install the package with development dependencies
	$(PYTHON) -m pip install -e ".[dev]"

test: ## Run the test suite
	$(PYTHON) -m pytest

lint: ## Lint the source tree
	$(PYTHON) -m ruff check src tests

format: ## Auto-format the source tree
	$(PYTHON) -m ruff format src tests

train-tiny: ## Train the tiny config (fast, for smoke testing)
	$(PYTHON) -m neurolm train --config configs/tiny.yaml

train-small: ## Train the small config
	$(PYTHON) -m neurolm train --config configs/small.yaml

generate: ## Sample from the small run (override PROMPT=...)
	$(PYTHON) -m neurolm generate --config configs/small.yaml \
		--checkpoint runs/small/final.npz --prompt "$(PROMPT)" --max-new-tokens 200

clean: ## Remove caches and training artifacts
	rm -rf runs .pytest_cache .ruff_cache build dist *.egg-info
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
