#!/usr/bin/env python3
"""
ArUco helpers: dictionary selection, detector params, and marker printing CLI.

Usage (module CLI):
    python -m src.utils.aruco_helpers --dict 4X4_50 --ids 0 1 2 3 --size 200 --border 20 --out markers
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


# ---- Dictionary helpers -----------------------------------------------------

DICT_ALIASES = {
    "4X4_50": cv2.aruco.DICT_4X4_50,
    "4X4_100": cv2.aruco.DICT_4X4_100,
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "4X4_1000": cv2.aruco.DICT_4X4_1000,
    "5X5_50": cv2.aruco.DICT_5X5_50,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "5X5_1000": cv2.aruco.DICT_5X5_1000,
    "6X6_50": cv2.aruco.DICT_6X6_50,
    "6X6_100": cv2.aruco.DICT_6X6_100,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "6X6_1000": cv2.aruco.DICT_6X6_1000,
    "7X7_50": cv2.aruco.DICT_7X7_50,
    "7X7_100": cv2.aruco.DICT_7X7_100,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
    "APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def get_dictionary(name: str) -> "cv2.aruco_Dictionary":
    key = name.upper().replace("DICT_", "")
    if key not in DICT_ALIASES:
        valid = ", ".join(sorted(DICT_ALIASES.keys()))
        raise ValueError(f"Unknown dictionary '{name}'. Valid options: {valid}")
    return cv2.aruco.getPredefinedDictionary(DICT_ALIASES[key])


# ---- Detector parameters ----------------------------------------------------


def create_detector_params() -> "cv2.aruco_DetectorParameters":
    params = cv2.aruco.DetectorParameters()
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 23
    params.adaptiveThreshWinSizeStep = 10
    params.adaptiveThreshConstant = 7

    params.minMarkerPerimeterRate = 0.02
    params.maxMarkerPerimeterRate = 4.0
    params.polygonalApproxAccuracyRate = 0.03

    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMaxIterations = 50
    params.cornerRefinementMinAccuracy = 0.01

    params.minDistanceToBorder = 3
    return params


# ---- Marker printing --------------------------------------------------------


def _draw_marker_image(dictionary: "cv2.aruco_Dictionary", marker_id: int, size_px: int) -> np.ndarray:
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(dictionary, marker_id, size_px)
    else:
        img = np.zeros((size_px, size_px), dtype=np.uint8)
        cv2.aruco.drawMarker(dictionary, marker_id, size_px, img, 1)
    return img


def print_markers(dict_name: str, ids: Iterable[int], size_px: int, out_dir: Path, border_px: int = 20) -> List[Path]:
    """
    Save each requested marker ID as a PNG in out_dir, with a white border (quiet zone).
    Returns list of saved file paths.
    """
    dictionary = get_dictionary(dict_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: List[Path] = []

    for mid in ids:
        core = _draw_marker_image(dictionary, int(mid), int(size_px))  # grayscale 0/255
        # Add a white quiet zone around the marker
        img = cv2.copyMakeBorder(
            core,
            border_px,
            border_px,
            border_px,
            border_px,
            borderType=cv2.BORDER_CONSTANT,
            value=255,  # white margin
        )
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        out_path = out_dir / f"marker_{mid}.png"
        if not cv2.imwrite(str(out_path), img_bgr):
            raise RuntimeError(f"Failed to write {out_path}")
        print(f"Saved {out_path}")
        saved.append(out_path)

    return saved


# ---- CLI --------------------------------------------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArUco marker printer CLI")
    p.add_argument("--dict", default="4X4_50", help="Dictionary name (e.g., 4X4_50, 5X5_100, APRILTAG_36h11)")
    p.add_argument("--ids", type=int, nargs="+", required=True, help="Marker IDs to generate (e.g., 0 1 2 3)")
    p.add_argument("--size", type=int, default=200, help="Marker size in pixels (marker square only, not counting border)")
    p.add_argument("--border", type=int, default=20, help="White border (quiet zone) in pixels (default: 20)")
    p.add_argument("--out", type=Path, default=Path("markers"), help="Output directory (default: markers)")
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(argv or sys.argv[1:])
    try:
        print_markers(ns.__dict__["dict"], ns.ids, ns.size, ns.out, border_px=int(ns.border))
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
