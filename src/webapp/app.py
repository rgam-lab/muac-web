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

import cv2
import numpy as np
from flask import Flask, make_response, render_template, request, Blueprint

from src.utils.aruco_helpers import create_detector_params, get_dictionary

# --- NEW: ChArUco detection/annotation helpers ---
from src.utils.charuco_helpers import (
    CharucoConfig,
    detect_charuco,
    draw_charuco,
    estimate_scale_m_per_px_from_charuco,
)

def detect_charuco_and_annotate(bgr: np.ndarray,
                                dict_name: str,
                                squares_x: int, squares_y: int,
                                square_m: float, marker_m: float,
                                p1: tuple[int,int] | None,
                                p2: tuple[int,int] | None) -> np.ndarray:
    """
    Detect ChArUco, draw board & (optional) distance between p1 and p2 in mm.
    Uses global scale estimate from the board.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cfg = CharucoConfig(
        squares_x=squares_x, squares_y=squares_y,
        square_length_m=square_m, marker_length_m=marker_m,
        dict_name=dict_name,
    )
    det = detect_charuco(gray, cfg)
    ann = draw_charuco(bgr.copy(), det)

    m_per_px = estimate_scale_m_per_px_from_charuco(det, cfg)
    if p1 and p2:
        # draw user segment + metric
        cv2.circle(ann, p1, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(ann, p2, 6, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.line(ann, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
        if m_per_px:
            dpx = float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))
            mm = dpx * m_per_px * 1000.0
            circ = np.pi * mm
            txt = f"D = {mm:.1f} mm   C = {circ:.1f} mm"
        else:
            txt = "No metric scale (need more ChArUco corners)"
        cv2.rectangle(ann, (0, 0), (ann.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(ann, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2, cv2.LINE_AA)
    else:
        # header: scale info
        cv2.rectangle(ann, (0, 0), (ann.shape[1], 28), (0, 0, 0), -1)
        msg = f"Scale: {m_per_px*1000:.3f} mm/px" if m_per_px else "Scale: unavailable"
        cv2.putText(ann, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 220, 255), 1, cv2.LINE_AA)

    return ann

# Web UI: enhancement tuner (client-only)
tuner_bp = Blueprint("tuner", __name__, template_folder="templates", static_folder="static")

@tuner_bp.route("/tuner")
def tuner():
    return render_template("tuner.html")

app = Flask(__name__)
app.register_blueprint(tuner_bp, url_prefix="")

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

# --- NEW ROUTE: /charuco_detect ---
@app.post("/charuco_detect")
def charuco_detect():
    """
    Expects raw PNG/JPEG frame bytes; query params:
      dict=4X4_50&sx=5&sy=7&sq=0.02&mk=0.014
      &p1=x,y&p2=x,y   (optional)
    Returns annotated PNG.
    """
    if not request.data:
        return ("No data", 400)

    # Decode frame
    nparr = np.frombuffer(request.data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return ("Bad image", 400)

    # Params
    dict_name = request.args.get("dict", "4X4_50")
    try:
        sx = int(request.args.get("sx", 5))
        sy = int(request.args.get("sy", 7))
        sq = float(request.args.get("sq", 0.02))
        mk = float(request.args.get("mk", 0.014))
    except Exception:
        return ("Bad board params", 400)

    def parse_pt(key: str):
        s = request.args.get(key)
        if not s:
            return None
        x, y = s.split(",")
        return (int(float(x)), int(float(y)))

    p1 = parse_pt("p1")
    p2 = parse_pt("p2")

    ann = detect_charuco_and_annotate(img, dict_name, sx, sy, sq, mk, p1, p2)

    ok, out = cv2.imencode(".png", ann)
    if not ok:
        return ("Encode failed", 500)
    resp = make_response(out.tobytes())
    resp.headers["Content-Type"] = "image/png"
    resp.headers["Cache-Control"] = "no-store"
    return resp

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
