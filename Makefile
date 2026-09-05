# # Define goinfre dynamically from the current user
# GOINFRE_DIR := /goinfre/$(USER)

# # Route uv cache/package downloads to goinfre
# export UV_CACHE_DIR := $(GOINFRE_DIR)/.uv_cache

# # Route uv-managed Python installations to goinfre
# export UV_PYTHON_INSTALL_DIR := $(GOINFRE_DIR)/.uv_python

# # Route the project's virtual environment to goinfre
# export UV_PROJECT_ENVIRONMENT := $(GOINFRE_DIR)/call_me_maybe_venv

# # Route Hugging Face model/dataset downloads to goinfre
# export HF_HOME := $(GOINFRE_DIR)/.huggingface_cache

.PHONY: install run debug clean lint lint-strict

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +

lint:
	uv run mypy src/ --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
	uv run flake8 src/

lint-strict:
	uv run mypy src/ --strict
	uv run flake8 src/