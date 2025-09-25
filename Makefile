# Simple helpers for venv, formatting, linting, and env check.
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff

.PHONY: venv fmt lint check-env

venv:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt

fmt:
	$(RUFF) check --fix .
	$(RUFF) format .

lint:
	$(RUFF) check .

check-env:
	$(PY) scripts/check_env.py
