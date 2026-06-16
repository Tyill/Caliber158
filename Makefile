# Caliber158 — project commands (source of truth for agents and CI).
# Requires: pixi (https://pixi.sh/). Env: copy .env.example → .env

.DEFAULT_GOAL := help

PIXI := pixi run
MOJO := $(PIXI) mojo

.PHONY: help install setup setup-python build check test test-grad test-grad-gpu smoke smoke-cpu smoke-cuda \
	info extract train train-cpu train-cuda clean

help: ## Show targets
	@printf "Caliber158 — common targets (run from repo root):\n\n"
	@grep -E '^[a-zA-Z0-9_.-]+:.*##' $(MAKEFILE_LIST) | \
		sed 's/:.*## /  /' | sort

install: ## Install pixi env (mojo + max)
	pixi install

setup: setup-python ## Alias: Python venv for teacher extract

setup-python: ## Create project .venv + PyTorch (CALIBER158_TORCH)
	$(PIXI) setup-python

build: ## Compile main.mojo; fail on compiler warnings
	@tmp=$$(mktemp); \
	$(MOJO) build main.mojo > "$$tmp" 2>&1; ec=$$?; \
	cat "$$tmp"; \
	if [ $$ec -ne 0 ]; then rm -f "$$tmp"; exit $$ec; fi; \
	if rg -q "warning:" "$$tmp"; then \
		echo "error: mojo build produced warnings (zero-warning policy)" >&2; \
		rm -f "$$tmp"; exit 1; \
	fi; \
	rm -f "$$tmp"

check: build ## Narrow check after Mojo edits (alias for build)

test-grad: build ## Regression: batched grads vs reference (< 1e-5)
	$(PIXI) test-grad

test-grad-gpu: build ## GPU backward vs CPU batch (< 1e-5); needs CUDA runtime
	$(PIXI) test-grad-gpu

smoke: build ## Quick synthetic train (device from CALIBER158_DEVICE)
	$(PIXI) smoke

smoke-cpu: build ## Smoke on CPU batch/GPU-off path
	CALIBER158_DEVICE=cpu $(PIXI) smoke

smoke-cuda: build ## Smoke on CUDA student path (needs NVIDIA GPU at build+run)
	CALIBER158_DEVICE=cuda $(PIXI) smoke

test: build test-grad smoke ## Full gate before commit (see no-commit-without-green-test)

info: ## Print env / CLI summary
	$(PIXI) info

extract: ## Teacher dataset → data/chains/*.bin (PyTorch CUDA/CPU)
	$(PIXI) extract

train: build ## Student train on CALIBER158_DATASET
	$(PIXI) train

train-cpu: build ## Student train forced to CPU
	CALIBER158_DEVICE=cpu $(PIXI) train

train-cuda: build ## Student train on CUDA (needs NVIDIA GPU at build+run)
	CALIBER158_DEVICE=cuda $(PIXI) train

clean: ## Remove local build artifacts
	rm -rf main.o main __pycache__ python/__pycache__ .mojo-cache
