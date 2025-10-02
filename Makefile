# Simple helpers for venv, formatting, linting, and env check.
VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip
RUFF := $(VENV)/bin/ruff
HOSTNAME := nominally-fleet-camel.ngrok-free.app
PORT := 80

.PHONY: venv fmt lint check-env run-webcam run-webapp host test \
	conda-create conda-activate dc-up dc-down dc-logs dc-rebuild \
	print-charuco print-charuco-pdf run-charuco-image run-charuco-webcam \
	tune-cam

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

# --- ChArUco board generation ---

# PNG (default 5x7, 20 mm squares, 14 mm markers)
print-charuco:
	$(PY) -c "from pathlib import Path; from src.utils.charuco_helpers import CharucoConfig, save_board_png; \
cfg=CharucoConfig(squares_x=5,squares_y=7,square_length_m=0.02,marker_length_m=0.014,dict_name='4X4_50'); \
save_board_png(cfg, Path('markers/charuco_5x7.png'))" && ls -lah markers/charuco_5x7.png

# US Letter PDF (true scale; print at 100% / 'Actual Size')
print-charuco-pdf:
	$(PY) -c "from pathlib import Path; from src.utils.charuco_helpers import CharucoConfig, save_board_pdf; \
cfg=CharucoConfig(squares_x=5,squares_y=7,square_length_m=0.02,marker_length_m=0.014,dict_name='4X4_50'); \
save_board_pdf(cfg, Path('markers/charuco_5x7_letter.pdf'))" && ls -lah markers/charuco_5x7_letter.pdf

# Detect ChArUco in an image and estimate metric scale
CHARUCO_IMAGE ?= markers/charuco_5x7.png
run-charuco-image:
	$(PY) src/charuco_demo_image.py --image $(CHARUCO_IMAGE) --squares-x 5 --squares-y 7 --square-m 0.04 --marker-m 0.028 --dict 4X4_50 --out out/charuco_annot.png

run-charuco-webcam:
	$(PY) src/charuco_demo_webcam.py --dict 4X4_50 --squares-x 5 --squares-y 7 --square-m 0.04 --marker-m 0.028

tune-cam:
	$(PY) src/tools/cam_enhance_tuner.py --camera-index 0 --width 1280 --height 720