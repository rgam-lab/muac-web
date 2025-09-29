#!/usr/bin/env python3
"""
ChArUco helpers:
- Board creation & PNG export
- Detection and optional pose
- Metric scale estimation (meters-per-pixel) from ChArUco geometry

This is the first pass toward arm circumference measurement:
we robustly get a local pixel scale from the detected board, so downstream
code can convert pixel distances (arm diameter) to meters and circumference.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .aruco_helpers import get_dictionary, create_detector_params


@dataclass
class CharucoConfig:
    squares_x: int                 # number of chessboard squares along X
    squares_y: int                 # number of chessboard squares along Y
    square_length_m: float         # square side length in meters (edge-to-edge)
    marker_length_m: float         # ArUco marker side length in meters
    dict_name: str = "4X4_50"


def create_board(cfg: CharucoConfig) -> "cv2.aruco_CharucoBoard":
    dictionary = get_dictionary(cfg.dict_name)
    board = cv2.aruco.CharucoBoard(
        (cfg.squares_x, cfg.squares_y),
        cfg.square_length_m,
        cfg.marker_length_m,
        dictionary,
    )
    return board


def save_board_png(
    cfg: CharucoConfig,
    out_path: Path,
    dpi: int = 300,
    border_bits: int = 1,
    size_pixels: Optional[Tuple[int, int]] = None,
) -> Path:
    """
    Render and save a printable ChArUco board.
    - If size_pixels is None, canvas size is inferred from squares & lengths (not to scale in pixels).
    - Use the generated PNG inside documents; print at 100% scaling so the physical square length matches.
    """
    board = create_board(cfg)
    if size_pixels is None:
        size_pixels = (2000, 1400)  # a reasonable default canvas
    img = board.generateImage(size_pixels, marginSize=border_bits, borderBits=border_bits)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), img)
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")
    return out_path


@dataclass
class CharucoDetection:
    charuco_corners: Optional[np.ndarray]  # Nx1x2 float32 image points
    charuco_ids: Optional[np.ndarray]      # Nx1 int32 IDs (chessboard corner IDs)
    aruco_corners: List[np.ndarray]
    aruco_ids: Optional[np.ndarray]
    image_size: Tuple[int, int]            # (w, h)


def detect_charuco(
    gray: np.ndarray,
    cfg: CharucoConfig,
) -> CharucoDetection:
    """
    Detect ArUco, then refine to ChArUco corners.
    Returns both sets and the image size.
    """
    dictionary = get_dictionary(cfg.dict_name)
    params = create_detector_params()

    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dictionary, params)
        aruco_corners, aruco_ids, _rej = det.detectMarkers(gray)
    else:
        aruco_corners, aruco_ids, _rej = cv2.aruco.detectMarkers(gray, dictionary, parameters=params)

    h, w = gray.shape[:2]
    if aruco_ids is None or len(aruco_ids) == 0:
        return CharucoDetection(None, None, aruco_corners, None, (w, h))

    # Interpolate ChArUco corners (chessboard corners) from ArUco detections
    board = create_board(cfg)
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        aruco_corners, aruco_ids, gray, board
    )

    if retval is None or retval < 4:  # need a few corners at least
        return CharucoDetection(None, None, aruco_corners, aruco_ids, (w, h))

    return CharucoDetection(charuco_corners, charuco_ids, aruco_corners, aruco_ids, (w, h))


def draw_charuco(
    image_bgr: np.ndarray,
    det: CharucoDetection,
) -> np.ndarray:
    """Overlay ArUco markers and ChArUco corners/IDs."""
    out = image_bgr.copy()
    if det.aruco_ids is not None and len(det.aruco_ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, det.aruco_corners, det.aruco_ids)
    if det.charuco_corners is not None and det.charuco_ids is not None:
        # draw ChArUco corners as small circles + ids
        for (pt,), (cid,) in zip(det.charuco_corners, det.charuco_ids):
            p = tuple(int(x) for x in pt)
            cv2.circle(out, p, 4, (0, 255, 0), -1)
            cv2.putText(out, str(int(cid)), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1, cv2.LINE_AA)
    return out


def estimate_scale_m_per_px_from_charuco(
    det: CharucoDetection,
    cfg: CharucoConfig,
) -> Optional[float]:
    """
    Estimate a global meters-per-pixel scale from ChArUco geometry,
    *without* needing camera intrinsics (first pass).

    Idea: The chessboard square side in the image (in pixels) divided into the
    known physical square length (meters) → meters per pixel.
    We approximate by computing distances between *adjacent* ChArUco corners
    that differ by exactly one step in the grid, then take the median.

    Notes:
    - This is a 1st-order approximation; perspective causes scale to vary across
      the image. Good enough as a starting point; we’ll refine using pose later.
    """
    if det.charuco_corners is None or det.charuco_ids is None:
        return None

    # Map ID -> 2D point
    pts = {int(i[0]): c[0] for c, i in zip(det.charuco_corners, det.charuco_ids)}

    # ChArUco corner IDs are laid out row-major across the inner chessboard intersections.
    # We can estimate adjacency by checking +/-1 and +/-squares_x offsets.
    sx = cfg.squares_x
    neighbors = []
    for cid, p in pts.items():
        for step in (1, sx):
            for sign in (-1, 1):
                nid = cid + sign * step
                if nid in pts:
                    q = pts[nid]
                    neighbors.append(np.linalg.norm(p - q))

    if not neighbors:
        return None

    px_per_square_candidates = [d for d in neighbors]  # distances in pixels between adjacent corners
    # Between adjacent chessboard corners, the physical distance is exactly square_length_m
    px_per_square = np.median(px_per_square_candidates)
    if px_per_square <= 0:
        return None

    m_per_px = cfg.square_length_m / px_per_square
    return float(m_per_px)
