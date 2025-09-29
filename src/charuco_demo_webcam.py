#!/usr/bin/env python3
"""
ChArUco webcam demo with live metric scale and simple arm diameter measurement.

Controls (GUI mode):
  - Left click: choose two points across the arm (diameter endpoints)
  - 'c'       : clear measurement points
  - 's'       : save current annotated frame to --save-dir (if provided)
  - 'q' or ESC: quit

CLI examples:
  GUI with defaults:
    python src/charuco_demo_webcam.py --dict 4X4_50 --squares-x 5 --squares-y 7 --square-m 0.02 --marker-m 0.014

  Headless logging (no GUI) + periodic frame saves:
    python src/charuco_demo_webcam.py --no-gui --save-dir out/ --fps 30
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from utils.camera import open_camera
from utils.charuco_helpers import (
    CharucoConfig,
    detect_charuco,
    draw_charuco,
    estimate_scale_m_per_px_from_charuco,
)

Point = Tuple[int, int]


def _put_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_measurement_overlay(frame: np.ndarray, p1: Optional[Point], p2: Optional[Point], m_per_px: Optional[float]) -> None:
    h, w = frame.shape[:2]
    # Instruction band
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    _put_text(frame, "Click two points across the arm | 'c' clear | 's' save | 'q' quit", (8, 20))

    if p1:
        cv2.circle(frame, p1, 5, (0, 255, 255), -1)
    if p2:
        cv2.circle(frame, p2, 5, (0, 255, 255), -1)
    if p1 and p2:
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
        px_dist = float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))
        if m_per_px:
            meters = px_dist * m_per_px
            mm = meters * 1000.0
            circ_mm = np.pi * mm
            _put_text(frame, f"D = {mm:.1f} mm  |  C = {circ_mm:.1f} mm", (8, h - 12), (0, 255, 0))
        else:
            _put_text(frame, f"D = {px_dist:.1f} px (no metric scale)", (8, h - 12), (0, 255, 255))


def main() -> int:
    ap = argparse.ArgumentParser(description="ChArUco webcam demo with live scale and arm diameter")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dict", default="4X4_50")
    ap.add_argument("--squares-x", type=int, default=5)
    ap.add_argument("--squares-y", type=int, default=7)
    ap.add_argument("--square-m", type=float, default=0.02, help="Square size (m)")
    ap.add_argument("--marker-m", type=float, default=0.014, help="Marker size (m)")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--scale-window", type=int, default=15, help="Frames in running median for scale smoothing")
    args = ap.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    cfg = CharucoConfig(
        squares_x=args.squares_x,
        squares_y=args.squares_y,
        square_length_m=args.square_m,
        marker_length_m=args.marker_m,
        dict_name=args.dict,
    )

    # Open camera
    cap, _fourcc = open_camera(args.camera_index, args.width, args.height, args.fps)

    # For GUI measurement clicks
    win = None if args.no_gui else "ChArUco Webcam"
    click_pts: List[Point] = []

    # Scale smoothing
    scale_hist: deque[float] = deque(maxlen=max(3, int(args.scale_window)))

    # Mouse callback (GUI)
    def on_mouse(event, x, y, flags, userdata):
        nonlocal click_pts
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pts.append((x, y))
            if len(click_pts) > 2:
                click_pts = click_pts[-2:]

    if win:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win, on_mouse)

    i = 0
    last_save_ts = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detect_charuco(gray, cfg)
            m_per_px = estimate_scale_m_per_px_from_charuco(det, cfg)
            if m_per_px is not None and np.isfinite(m_per_px) and m_per_px > 0:
                scale_hist.append(float(m_per_px))
            smoothed = float(np.median(scale_hist)) if len(scale_hist) >= 3 else (m_per_px if m_per_px else None)

            # Draw detections & HUD
            vis = draw_charuco(frame, det)
            # Scale banner
            cv2.rectangle(vis, (0, 28), (vis.shape[1], 50), (0, 0, 0), -1)
            if smoothed:
                _put_text(vis, f"scale ~ {smoothed*1000:.3f} mm/px", (8, 44), (0, 255, 0))
            else:
                _put_text(vis, "scale unavailable (need board in view)", (8, 44), (0, 255, 255))

            # Measurement overlay
            p1 = click_pts[0] if len(click_pts) >= 1 else None
            p2 = click_pts[1] if len(click_pts) >= 2 else None
            _draw_measurement_overlay(vis, p1, p2, smoothed)

            if args.no_gui:
                # Log every second; optionally save frames periodically
                if i % args.fps == 0:
                    msg = f"[{time.strftime('%H:%M:%S')}] scale="
                    msg += f"{smoothed:.6f} m/px" if smoothed else "NA"
                    if p1 and p2 and smoothed:
                        px_dist = float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))
                        mm = px_dist * smoothed * 1000.0
                        circ_mm = np.pi * mm
                        msg += f" | D={mm:.1f}mm C={circ_mm:.1f}mm"
                    print(msg)
                if args.save_dir and (time.time() - last_save_ts) >= 1.0:
                    out_path = args.save_dir / f"charuco_{int(time.time())}.png"
                    cv2.imwrite(str(out_path), vis)
                    last_save_ts = time.time()
            else:
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):
                    break
                if key == ord('c'):
                    click_pts.clear()
                if key == ord('s') and args.save_dir:
                    out_path = args.save_dir / f"charuco_{int(time.time())}.png"
                    cv2.imwrite(str(out_path), vis)
                    print(f"Saved {out_path}")

            i += 1

    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
