# Simple helpers for venv, formatting, linting, and env check.
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
HOSTNAME := nominally-fleet-camel.ngrok-free.app
PORT := 80

.PHONY: venv fmt lint check-env run-webcam run-webapp host test conda-create conda-activate dc-up dc-down dc-logs dc-rebuild

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

run-webcam:
	$(PY) src/aruco_demo_webcam.py --dict 4X4_50

run-webapp:
	FLASK_APP=src/webapp/app.py $(PY) -m flask run --host 0.0.0.0 -p $(PORT)

host:
	http --hostname=$(HOSTNAME) $(PORT)

test:
	$(PY) -m pytest

conda-create:
	conda env create -f environment.yml

# Activation can't happen inside Make; run this in your shell:
# conda activate aruco-env
conda-activate:
	@echo "Run 'conda activate aruco-env' in your shell."

dc-up:
	docker compose up -d

dc-down:
	docker compose down

dc-logs:
	docker compose logs -f

dc-rebuild:
	docker compose build --no-cache