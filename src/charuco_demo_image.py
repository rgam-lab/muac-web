#!/usr/bin/env python3
"""
ChArUco image demo:
- Detect board
- Annotate image
- Estimate global meters-per-pixel scale (first pass)
- (Optional) measure distance between two points you provide

Examples:
    python src/charuco_demo_image.py --image samples/charuco.jpg \
      --squares-x 5 --squares-y 7 --square-m 0.02 --marker-m 0.014 \
      --dict 4X4_50 --out out/charuco_annot.png

    # Measure distance between two pixel points (x1,y1 x2,y2):
    python src/charuco_demo_image.py --image path.jpg \
      --squares-x 5 --squares-y 7 --square-m 0.02 --marker-m 0.014 \
      --measure 250,320 410,330
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from utils.charuco_helpers import (
    CharucoConfig,
    detect_charuco,
    draw_charuco,
    estimate_scale_m_per_px_from_charuco,  # re-exported for convenience
)


def _parse_point(s: str) -> Tuple[int, int]:
    x, y = s.split(",")
    return int(x), int(y)


def main() -> int:
    ap = argparse.ArgumentParser(description="ChArUco detection + scale estimation demo")
    ap.add_argument("--image", required=True, help="Path to input image")
    ap.add_argument("--dict", default="4X4_50", help="ArUco dictionary for the board (e.g., 4X4_50)")
    ap.add_argument("--squares-x", type=int, default=5, help="ChArUco squares along X")
    ap.add_argument("--squares-y", type=int, default=7, help="ChArUco squares along Y")
    ap.add_argument("--square-m", type=float, default=0.02, help="Square size in meters (edge-to-edge)")
    ap.add_argument("--marker-m", type=float, default=0.014, help="Marker size in meters (marker square side)")
    ap.add_argument("--out", type=Path, default=Path("out/charuco_annot.png"), help="Output annotated path")
    ap.add_argument("--measure", nargs=2, metavar=("x1,y1", "x2,y2"), help="Optional measure between two points (pixels)")
    args = ap.parse_args()

    img_path = Path(args.image)
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cfg = CharucoConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_m,
        marker_length_m=args.marker_m,
        dict_name=args.dict,
    )

    det = detect_charuco(gray, cfg)
    n_markers = 0 if det.aruco_ids is None else int(len(det.aruco_ids))
    n_charuco = 0 if det.charuco_ids is None else int(len(det.charuco_ids))
    print(f"ArUco markers: {n_markers}, ChArUco corners: {n_charuco}")

    m_per_px = estimate_scale_m_per_px_from_charuco(det, cfg)
    if m_per_px is None:
        print("Scale estimation: unavailable (need more ChArUco corners).")
    else:
        print(f"Estimated scale: {m_per_px:.6f} m/px  (~{1.0/m_per_px:.1f} px/m)")

    annot = draw_charuco(bgr, det)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), annot)
    print(f"Saved: {args.out.resolve()}")

    # Optional distance measure between two given pixel points
    if args.measure:
        p1 = _parse_point(args.measure[0])
        p2 = _parse_point(args.measure[1])
        px_dist = float(np.linalg.norm(np.array(p1, dtype=float) - np.array(p2, dtype=float)))
        if m_per_px is not None:
            meters = px_dist * m_per_px
            print(f"Distance {p1}→{p2}: {px_dist:.2f} px  ≈  {meters*1000:.2f} mm")
        else:
            print(f"Distance {p1}→{p2}: {px_dist:.2f} px  (no metric scale available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
