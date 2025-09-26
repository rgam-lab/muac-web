#!/usr/bin/env python3
"""
Minimal Flask web app for ArUco detection.
- Serves a simple page that uses getUserMedia to capture camera frames.
- Frames are POSTed to /detect as PNG; server returns annotated PNG.

Note:
- Mobile browsers often require HTTPS for camera (localhost on desktop is OK).
- For LAN/mobile testing, consider `flask run --host 0.0.0.0` and use HTTPS via a proxy if needed.
"""
from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from flask import Flask, render_template, request, make_response

from src.utils.aruco_helpers import get_dictionary, create_detector_params

app = Flask(__name__)


def detect_and_annotate(bgr: np.ndarray, dict_name: str = "4X4_50") -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    dictionary = get_dictionary(dict_name)
    params = create_detector_params()
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = det.detectMarkers(gray)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(bgr, corners, ids)
    return bgr


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/detect")
def detect():
    """
    Expects raw PNG/JPEG frame bytes in request.data.
    Returns annotated PNG.
    """
    if not request.data:
        return ("No data", 400)

    # Decode image bytes (works for PNG/JPEG)
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ("Bad image", 400)

    ann = detect_and_annotate(img, dict_name=request.args.get("dict", "4X4_50"))
    ok, out = cv2.imencode(".png", ann)
    if not ok:
        return ("Encode failed", 500)

    resp = make_response(out.tobytes())
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-store"
    return resp


if __name__ == "__main__":
    # Dev server
    app.run(debug=True)
