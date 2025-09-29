#!/usr/bin/env python3
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
    estimate_scale_m_per_px_from_charuco,  # keep as fallback
    estimate_homography_from_charuco,
    local_scale_m_per_px,
)

Point = Tuple[int, int]


def _put_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_measurement_overlay(frame: np.ndarray, p1: Optional[Point], p2: Optional[Point], m_per_px: Optional[float]) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    _put_text(frame, "Click two points across the arm | 'c' clear | 's' save | 'q' quit | 'a' toggle auto", (8, 20))
    if p1:
        cv2.circle(frame, p1, 5, (0, 255, 255), -1)
    if p2:
        cv2.circle(frame, p2, 5, (0, 255, 255), -1)
    if p1 and p2:
        cv2.line(frame, p1, p2, (0, 255, 0), 2)
        px_dist = float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))
        if m_per_px:
            mm = px_dist * m_per_px * 1000.0
            circ_mm = np.pi * mm
            _put_text(frame, f"D = {mm:.1f} mm  |  C = {circ_mm:.1f} mm", (8, h - 12), (0, 255, 0))
        else:
            _put_text(frame, f"D = {px_dist:.1f} px (no metric scale)", (8, h - 12), (0, 255, 255))


# ---- NEW: auto arm-band ROI helper -----------------------------------------

def _auto_band_diameter_px(gray: np.ndarray) -> Optional[Tuple[Point, Point, float]]:
    """
    Simple automatic diameter finder in a narrow horizontal stripe around the mid-height.
    Steps:
      - Use Sobel X gradients collapsed across a band to form a 1D profile along columns.
      - Find two strongest opposing edges with reasonable separation.
    Returns (p1, p2, diameter_px) or None if not reliable.
    """
    h, w = gray.shape[:2]
    band_h = max(6, h // 20)            # ~5% of image height
    y0 = h // 2 - band_h // 2
    roi = gray[y0:y0 + band_h, :]

    # Smooth then gradient along X
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    gx = cv2.Sobel(roi_blur, cv2.CV_32F, 1, 0, ksize=3)
    mag = np.abs(gx)

    # Collapse to 1D column response
    profile = mag.mean(axis=0)  # (w,)

    # Normalize & find peaks
    prof = profile.copy()
    if prof.max() <= 1e-6:
        return None
    prof /= prof.max()

    # Basic peak picking: top two peaks with min separation
    min_sep = max(20, w // 15)  # pixels
    idx_sorted = np.argsort(prof)[::-1]
    peaks: List[int] = []
    for idx in idx_sorted:
        if all(abs(idx - p) >= min_sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= 2:
            break
    if len(peaks) < 2:
        return None
    peaks.sort()
    x1, x2 = peaks[0], peaks[1]
    diameter_px = float(abs(x2 - x1))
    p1 = (x1, y0 + band_h // 2)
    p2 = (x2, y0 + band_h // 2)

    # Heuristic sanity: reject silly widths (too small/too big)
    if diameter_px < w * 0.05 or diameter_px > w * 0.9:
        return None
    return p1, p2, diameter_px


def main() -> int:
    ap = argparse.ArgumentParser(description="ChArUco webcam demo with pose-aware scale and auto arm band")
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
    ap.add_argument("--scale-window", type=int, default=15, help="Frames in running median for global scale smoothing")
    ap.add_argument("--auto", action="store_true", help="Start with auto band detection enabled")
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

    cap, _fourcc = open_camera(args.camera_index, args.width, args.height, args.fps)

    win = None if args.no_gui else "ChArUco Webcam"
    click_pts: List[Point] = []
    auto_on = bool(args.auto)

    # Global scale fallback smoothing (from old method)
    global_scale_hist: deque[float] = deque(maxlen=max(3, int(args.scale_window)))

    Hinv: Optional[np.ndarray] = None
    H_used = 0

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

            # Try homography (pose-aware local scale)
            H, Hinv_new, n_used = estimate_homography_from_charuco(det, cfg)
            if Hinv_new is not None:
                Hinv = Hinv_new
                H_used = n_used

            # Compute a global fallback scale for display if Hinv missing
            gscale = estimate_scale_m_per_px_from_charuco(det, cfg)
            if gscale and np.isfinite(gscale) and gscale > 0:
                global_scale_hist.append(float(gscale))
            gscale_smoothed = float(np.median(global_scale_hist)) if len(global_scale_hist) >= 3 else (gscale if gscale else None)

            # Visualization
            vis = draw_charuco(frame, det)

            # Status bars
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            if Hinv is not None:
                _put_text(vis, f"pose-aware scale ✓  (pts={H_used})", (8, 20), (0, 255, 0))
            else:
                _put_text(vis, "pose-aware scale ×  (need more ChArUco)", (8, 20), (0, 255, 255))
            if gscale_smoothed:
                _put_text(vis, f"global scale ~ {gscale_smoothed*1000:.3f} mm/px", (8, 40), (200, 200, 255))

            # ----- Auto band (optional) -----
            auto_result = None
            if auto_on:
                auto_result = _auto_band_diameter_px(gray)
                if auto_result:
                    p1, p2, diameter_px = auto_result
                    # local scale at midpoint (pose-aware if Hinv, else fallback)
                    mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
                    if Hinv is not None:
                        mpp = local_scale_m_per_px(Hinv, mid[0], mid[1])
                    else:
                        mpp = gscale_smoothed
                    if mpp:
                        mm = diameter_px * mpp * 1000.0
                        circ_mm = np.pi * mm
                        cv2.line(vis, p1, p2, (0, 180, 255), 2)
                        _put_text(vis, f"[auto] D={mm:.1f} mm  C={circ_mm:.1f} mm", (8, vis.shape[0]-32), (0, 180, 255))

            # ----- Click measurement (always available) -----
            p1 = click_pts[0] if len(click_pts) >= 1 else None
            p2 = click_pts[1] if len(click_pts) >= 2 else None
            # Prefer local pose-aware scale at the segment midpoint; fallback to global
            seg_mpp: Optional[float] = None
            if p1 and p2:
                mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
                if Hinv is not None:
                    seg_mpp = local_scale_m_per_px(Hinv, mid[0], mid[1])
                else:
                    seg_mpp = gscale_smoothed

            _draw_measurement_overlay(vis, p1, p2, seg_mpp)

            # GUI / headless flows
            if args.no_gui:
                if i % args.fps == 0:
                    msg = f"[{time.strftime('%H:%M:%S')}] "
                    msg += f"Hinv={'Y' if Hinv is not None else 'N'} "
                    if auto_result and (Hinv is not None or gscale_smoothed):
                        msg += "auto=OK "
                    if p1 and p2 and seg_mpp:
                        px_dist = float(np.linalg.norm(np.array(p1, float) - np.array(p2, float)))
                        mm = px_dist * seg_mpp * 1000.0
                        circ_mm = np.pi * mm
                        msg += f"| click D={mm:.1f}mm C={circ_mm:.1f}mm"
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
                if key == ord('a'):
                    auto_on = not auto_on

            i += 1

    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
