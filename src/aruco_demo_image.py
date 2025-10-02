#!/usr/bin/env python3
"""
ArUco image demo: detect markers in a still image, annotate, and optionally draw pose axes.

Examples:
    python src/aruco_demo_image.py --image samples/aruco_test.jpg --dict 4X4_50 --out out/annotated.png
    python src/aruco_demo_image.py --image path/to.jpg --dict 4X4_50 --marker-length 0.04 \
        --camera-matrix cam_k.json --dist-coeffs cam_d.json --out out/annotated.png

Expected JSON formats:
    cam_k.json:
        { "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]] }
    cam_d.json:
        { "dist_coeffs": [k1, k2, p1, p2, k3] }  # 4,5,8, or 12 values are accepted by OpenCV
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from utils.aruco_helpers import create_detector_params, get_dictionary


def _load_json(path: Optional[str]) -> Optional[dict]:
    if not path:
        return None
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_intrinsics(k_path: Optional[str], d_path: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    k_json = _load_json(k_path)
    d_json = _load_json(d_path)
    K = None
    D = None
    if k_json and "camera_matrix" in k_json:
        K = np.asarray(k_json["camera_matrix"], dtype=np.float32)
        if K.shape != (3, 3):
            raise ValueError(f"camera_matrix must be 3x3; got {K.shape}")
    if d_json and "dist_coeffs" in d_json:
        D = np.asarray(d_json["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
    return K, D


def _detect_markers(gray: np.ndarray, dict_name: str):
    dictionary = get_dictionary(dict_name)
    params = create_detector_params()

    # Newer API: ArucoDetector; fallback to detectMarkers if unavailable.
    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)
    return corners, ids, rejected


def _draw_annotations(
    img_bgr: np.ndarray,
    corners, ids,
    K: Optional[np.ndarray], D: Optional[np.ndarray],
    marker_length_m: float,
    draw_axes: bool
) -> np.ndarray:
    out = img_bgr.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)

        if draw_axes and K is not None and D is not None:
            # Pose per marker
            rvecs, tvecs, _objPoints = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length_m, K, D)
            for rvec, tvec in zip(rvecs, tvecs):
                cv2.drawFrameAxes(out, K, D, rvec, tvec, marker_length_m * 0.5)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Detect ArUco markers in an image and annotate the result.")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--dict", default="4X4_50", help="Dictionary name (e.g., 4X4_50, 5X5_100, APRILTAG_36h11)")
    ap.add_argument("--marker-length", type=float, default=0.04, help="Marker length in meters (for pose axes)")
    ap.add_argument("--camera-matrix", help="Path to JSON with key 'camera_matrix' (3x3)")
    ap.add_argument("--dist-coeffs", help="Path to JSON with key 'dist_coeffs' (length 4/5/8/12)")
    ap.add_argument("--out", type=Path, default=Path("out/annotated.png"), help="Output annotated image path")
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Input image not found: {img_path}")

    # Load image + grayscale
    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Detect
    corners, ids, rejected = _detect_markers(gray, args.dict)
    num_found = 0 if ids is None else int(len(ids))
    print(f"Markers found: {num_found}")

    # Intrinsics for pose/axes (optional)
    K, D = _load_intrinsics(args.camera_matrix, args.dist_coeffs)
    draw_axes = K is not None and D is not None

    # Draw
    annotated = _draw_annotations(img_bgr, corners, ids, K, D, float(args.marker_length), draw_axes)

    # Ensure output dir exists and save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), annotated):
        raise SystemExit(f"Failed to write output image: {args.out}")
    print(f"Saved: {args.out.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
