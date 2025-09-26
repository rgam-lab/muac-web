# ArUco Toolkit (Python + OpenCV)

Minimal computer-vision toolkit to detect, visualize, and generate ArUco markers.  
Intended as a clean, reproducible starting point for robotics, AR, and computer vision projects.

## Quickstart

```bash
make venv
source .venv/bin/activate

make check-env

# Run the webcam script locally
make run-webcam
# or host the webapp
make run-webapp

# To public host
make host
```

## Features (planned)

- Detect ArUco markers in images and webcam streams
- Generate and print ArUco markers
- Pose estimation with optional camera calibration
- Reproducible builds with venv, Conda, and Docker

