# ArUco Toolkit (Python + OpenCV)

Minimal computer-vision toolkit to detect, visualize, and generate ArUco markers.  
Intended as a clean, reproducible starting point for robotics, AR, and computer vision projects.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
pip install --upgrade pip
pip install -r requirements.txt
# Then run the environment check (will be added in the next step):
python scripts/check_env.py
```

## Features (planned)

- Detect ArUco markers in images and webcam streams
- Generate and print ArUco markers
- Pose estimation with optional camera calibration
- Reproducible builds with venv, Conda, and Docker

