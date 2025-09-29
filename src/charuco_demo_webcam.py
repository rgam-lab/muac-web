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
    estimate_scale_m_per_px_from_charuco,  # fallback
    estimate_homography_from_charuco,
    local_scale_m_per_px,
    project_board_polygon_px,
)

Point = Tuple[int, int]


def _put_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_measurement_overlay(frame: np.ndarray, p1: Optional[Point], p2: Optional[Point], m_per_px: Optional[float]) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    _put_text(frame, "Click 2 points | 'c' clear | 's' save | 'q' quit | 'a' auto | 'o' orient | arrows move band", (8, 20))
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


# ====== Guide-band geometry (orientation-aware) ======

def _rect_from_center_band(h: int, w: int, cy: int, cx: int, size: int, orient: str) -> Tuple[int, int, int, int]:
    if orient == 'v':
        x0 = max(0, cx - size // 2)
        x1 = min(w, x0 + size)
        return x0, 0, x1, h
    y0 = max(0, cy - size // 2)
    y1 = min(h, y0 + size)
    return 0, y0, w, y1


def _draw_guide_band(vis: np.ndarray, band_rect: Tuple[int, int, int, int], orient: str) -> None:
    x0, y0, x1, y1 = band_rect
    overlay = vis.copy()
    color = (0, 200, 255)
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
    alpha = 0.12
    cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
    msg = "Align upper arm in band (measures ⟂ to band) | 'o' toggles H/V"
    _put_text(vis, msg, (x0 + 8, max(16, y0 - 6) if orient == 'h' else 20), color)


def _mask_out_board(h: int, w: int, board_poly: Optional[np.ndarray]) -> np.ndarray:
    keep = np.ones((h, w), dtype=np.uint8)
    if board_poly is None:
        return keep
    board_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(board_mask, [board_poly.astype(np.int32)], 1)
    board_mask = cv2.dilate(board_mask, np.ones((7, 7), np.uint8), iterations=1)
    keep[board_mask == 1] = 0
    return keep

# ====== Band enhancement + brightness-cutoff auto ======

def _enhance_band(gray_band: np.ndarray, contrast_pct: int, sharp_pct: int, gamma: float) -> np.ndarray:
    """
    Strong, stable enhancement for low-texture limbs:
      - CLAHE for local contrast
      - Optional gamma
      - Unsharp mask (controllable sharpness)
      - Final linear contrast gain (alpha)
    """
    # 1) CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    x = clahe.apply(gray_band)

    # 2) Gamma (optional)
    if abs(gamma - 1.0) > 1e-3:
        x = np.clip((x.astype(np.float32) / 255.0) ** (1.0 / gamma) * 255.0, 0, 255).astype(np.uint8)

    # 3) Unsharp mask
    blur = cv2.GaussianBlur(x, (0, 0), sigmaX=1.2)
    sharp_amount = max(0.0, min(2.0, sharp_pct / 100.0 * 1.6))  # 0..~1.6
    x = cv2.addWeighted(x, 1 + sharp_amount, blur, -sharp_amount, 0)

    # 4) Linear contrast
    alpha = 1.0 + (contrast_pct / 100.0) * 1.2  # up to ~2.2
    x = cv2.convertScaleAbs(x, alpha=alpha, beta=0)

    return x


def _skin_mask_rgb(img_bgr: np.ndarray) -> np.ndarray:
    """
    Broad, inclusive skin-ish mask combining YCrCb and HSV conditions.
    Returns uint8 mask in {0,1}. NOT exclusive; background may leak through,
    but it's OK because we later intersect with band + board 'keep' mask.

    References adapted from multiple open CV recipes, widened to cover a wide gamut.
    """
    # YCrCb path (stable under illumination)
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    # Very broad Cr/Cb windows (covering very light to very dark skin)
    mask_ycrcb = (
        (Cr >= 120) & (Cr <= 180) &   # Cr: 120–180 (broad)
        (Cb >= 70)  & (Cb <= 150)     # Cb: 70–150  (broad)
    )

    # HSV path (handles hue & saturation)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    # Two HSV windows to capture varied lighting:
    mask_hsv1 = (H <= 25) & (S >= 30) & (V >= 20)    # warm/low-H, min saturation/value
    mask_hsv2 = (H >= 0) & (H <= 50) & (S >= 15) & (V >= 15)  # even wider, very permissive
    mask_hsv = mask_hsv1 | mask_hsv2

    # Union of YCrCb & HSV
    mask = (mask_ycrcb | mask_hsv).astype(np.uint8)

    # Morphological clean-up (small holes/noise)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask  # {0,1}


def _auto_diameter_binary(
    gray: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask: np.ndarray,
    orient: str,
    contrast_pct: int,
    sharp_pct: int,
    gamma: float,
    full_bgr: Optional[np.ndarray] = None,
    use_skin: bool = False,
    skin_debug_draw: Optional[np.ndarray] = None,
) -> Optional[Tuple[Point, Point, float]]:
    """
    Brightness-cutoff (binary) auto with optional skin prefilter:
      - OPTIONAL: intersect with skin mask (YCbCr+HSV union) to bias toward arm pixels.
      - Enhance band
      - Otsu threshold
      - Morph cleanup
      - 1D projection perpendicular to band; pick two robust borders
    """
    x0, y0, x1, y1 = band_rect
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None

    km = keep_mask[y0:y1, x0:x1]  # 1=keep, 0=remove (board)
    band_enh = _enhance_band(band, contrast_pct, sharp_pct, gamma)

    # Start with all-ones mask inside the band
    pre_mask = np.ones_like(band, dtype=np.uint8)

    # Optional: skin filter inside the band (prefer arm)
    if use_skin and full_bgr is not None:
        band_bgr = full_bgr[y0:y1, x0:x1]
        sm = _skin_mask_rgb(band_bgr)  # {0,1}
        pre_mask = sm
        if skin_debug_draw is not None:
            # Draw a tiny 80x80 thumbnail of the skin mask in the top-right for debugging
            thumb = (sm * 255)
            th = min(80, thumb.shape[0])
            tw = min(80, thumb.shape[1])
            thumb = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_NEAREST)
            skin_debug_draw[4:4+th, -tw-4:-4] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

    # Threshold with Otsu
    _, otsu = cv2.threshold(band_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Combine: remove board area, require skin (if enabled)
    combo = (otsu > 0).astype(np.uint8)
    combo[km == 0] = 0
    combo &= pre_mask  # if skin used, this keeps only skin-ish foreground

    # Clean silhouette
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(combo * 255, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = (mask > 0).astype(np.uint8)

    # Project perpendicular to band to get diameter
    if orient == 'h':
        profile = mask.mean(axis=1).astype(np.float32)   # rows
        axis_len = profile.shape[0]
        if axis_len < 6 or profile.max() < 0.08:
            return None
        min_sep = max(20, axis_len // 10)
        idx_sorted = np.argsort(profile)[::-1]
        picks: List[int] = []
        for idx in idx_sorted:
            if all(abs(int(idx) - p) >= min_sep for p in picks):
                picks.append(int(idx))
            if len(picks) >= 2:
                break
        if len(picks) < 2:
            return None
        picks.sort()
        yT, yB = picks[0], picks[1]
        height = yB - yT
        if height < axis_len * 0.06 or height > axis_len * 0.98:
            return None
        xC = (x0 + x1) // 2
        return (xC, y0 + yT), (xC, y0 + yB), float(height)

    # orient == 'v'
    profile = mask.mean(axis=0).astype(np.float32)       # cols
    axis_len = profile.shape[0]
    if axis_len < 6 or profile.max() < 0.08:
        return None
    min_sep = max(20, axis_len // 10)
    idx_sorted = np.argsort(profile)[::-1]
    picks: List[int] = []
    for idx in idx_sorted:
        if all(abs(int(idx) - p) >= min_sep for p in picks):
            picks.append(int(idx))
        if len(picks) >= 2:
            break
    if len(picks) < 2:
        return None
    picks.sort()
    xL, xR = picks[0], picks[1]
    width = xR - xL
    if width < axis_len * 0.06 or width > axis_len * 0.98:
        return None
    yC = (y0 + y1) // 2
    return (x0 + xL, yC), (x0 + xR, yC), float(width)



# ====== Auto diameter (perpendicular to band) ======

def _auto_diameter_px(
    gray: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask: np.ndarray,
    orient: str,
) -> Optional[Tuple[Point, Point, float]]:
    """
    Auto diameter in the band, measuring PERPENDICULAR to band orientation:
      - orient='h' → measure vertical diameter (top↕bottom) using row profile.
      - orient='v' → measure horizontal diameter (left↔right) using column profile.
    """
    x0, y0, x1, y1 = band_rect
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None

    km = keep_mask[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(band, (5, 5), 0)

    if orient == 'h':
        # Measure vertical: use Sobel Y/Canny, then ROW profile
        edges = cv2.Canny(blur, 40, 120)
        edges[km == 0] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        profile = edges.mean(axis=1)  # rows → vertical edges density
        axis_len = profile.shape[0]
        min_sep = max(24, axis_len // 12)
        if profile.max() < 5:
            return None
        prof = (profile / (profile.max() + 1e-6)).astype(np.float32)
        idx_sorted = np.argsort(prof)[::-1]
        peaks: List[int] = []
        for idx in idx_sorted:
            if all(abs(int(idx) - p) >= min_sep for p in peaks):
                peaks.append(int(idx))
            if len(peaks) >= 2:
                break
        if len(peaks) < 2:
            return None
        peaks.sort()
        yT, yB = peaks[0], peaks[1]
        strength_balance = min(prof[yT], prof[yB]) / (max(prof[yT], prof[yB]) + 1e-6)
        height = yB - yT
        if strength_balance < 0.4 or height < axis_len * 0.06 or height > axis_len * 0.95:
            return None
        xC = (x0 + x1) // 2
        p1 = (xC, y0 + yT)
        p2 = (xC, y0 + yB)
        return p1, p2, float(height)

    # orient == 'v' → measure horizontal (left/right)
    edges = cv2.Canny(blur, 40, 120)
    edges[km == 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    profile = edges.mean(axis=0)  # columns
    axis_len = profile.shape[0]
    min_sep = max(24, axis_len // 12)
    if profile.max() < 5:
        return None
    prof = (profile / (profile.max() + 1e-6)).astype(np.float32)
    idx_sorted = np.argsort(prof)[::-1]
    peaks: List[int] = []
    for idx in idx_sorted:
        if all(abs(int(idx) - p) >= min_sep for p in peaks):
            peaks.append(int(idx))
        if len(peaks) >= 2:
            break
    if len(peaks) < 2:
        return None
    peaks.sort()
    xL, xR = peaks[0], peaks[1]
    strength_balance = min(prof[xL], prof[xR]) / (max(prof[xL], prof[xR]) + 1e-6)
    width = xR - xL
    if strength_balance < 0.4 or width < axis_len * 0.06 or width > axis_len * 0.95:
        return None
    yC = (y0 + y1) // 2
    p1 = (x0 + xL, yC)
    p2 = (x0 + xR, yC)
    return p1, p2, float(width)


def main() -> int:
    ap = argparse.ArgumentParser(description="ChArUco webcam demo with pose-aware scale, guide band, and auto arm detection")
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
    ap.add_argument("--scale-window", type=int, default=15)
    ap.add_argument("--auto", action="store_true", help="Start with auto band detection enabled")
    ap.add_argument("--band-h", type=int, default=64, help="Horizontal band height (pixels) or vertical band width (pixels)")
    ap.add_argument("--band-y", type=int, default=-1, help="Horizontal band center y (default: mid-height)")
    ap.add_argument("--band-x", type=int, default=-1, help="Vertical band center x (default: mid-width)")
    ap.add_argument("--band-orient", choices=["h", "v"], default="h", help="Band orientation: 'h' horizontal (measure vertical), 'v' vertical (measure horizontal)")
    ap.add_argument("--auto-contrast", type=int, default=100, help="Auto band contrast boost percent (0-100)")
    ap.add_argument("--auto-sharp", type=int, default=100, help="Auto band sharpness boost percent (0-100)")
    ap.add_argument("--auto-gamma", type=float, default=1.0, help="Auto band gamma (e.g., 0.9 darker, 1.1 brighter)")
    ap.add_argument("--auto-method", choices=["bin", "edge"], default="bin", help="Auto method: 'bin' brightness cutoff (default) or 'edge' gradient peaks")
    ap.add_argument("--skin", action="store_true", help="Enable broad skin prefilter in band (YCbCr+HSV union)")
    ap.add_argument("--skin-debug", action="store_true", help="Show a tiny skin mask indicator in the overlay")


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

    global_scale_hist: deque[float] = deque(maxlen=max(3, int(args.scale_window)))

    H: Optional[np.ndarray] = None
    Hinv: Optional[np.ndarray] = None
    H_used = 0

    # guide band state
    h0, w0 = args.height, args.width
    band_cy = args.band_y if args.band_y >= 0 else h0 // 2
    band_cx = args.band_x if args.band_x >= 0 else w0 // 2
    band_size = max(24, int(args.band_h))
    band_orient = args.band_orient  # 'h' or 'v'

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

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detect_charuco(gray, cfg)

            # Pose-aware H + inverse
            H_new, Hinv_new, n_used = estimate_homography_from_charuco(det, cfg)
            if H_new is not None and Hinv_new is not None:
                H, Hinv, H_used = H_new, Hinv_new, n_used

            # Fallback global scale
            gscale = estimate_scale_m_per_px_from_charuco(det, cfg)
            if gscale and np.isfinite(gscale) and gscale > 0:
                global_scale_hist.append(float(gscale))
            gscale_smoothed = float(np.median(global_scale_hist)) if len(global_scale_hist) >= 3 else (gscale if gscale else None)

            # Board mask
            board_poly = project_board_polygon_px(H, cfg)
            keep_mask = _mask_out_board(h, w, board_poly)

            # Visualization base
            vis = draw_charuco(frame, det)

            # Status
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 50), (0, 0, 0), -1)
            if Hinv is not None:
                _put_text(vis, f"pose-aware scale ✓ (pts={H_used})", (8, 20), (0, 255, 0))
            else:
                _put_text(vis, "pose-aware scale × (need more ChArUco)", (8, 20), (0, 255, 255))
            if gscale_smoothed:
                _put_text(vis, f"global ~ {gscale_smoothed*1000:.3f} mm/px", (8, 40), (200, 200, 255))

            # Guide band (orientation-aware)
            band_rect = _rect_from_center_band(h, w, band_cy, band_cx, band_size, band_orient)
            _draw_guide_band(vis, band_rect, band_orient)

            # Auto detection in band
            auto_result = None
            if auto_on:
                if args.auto_method == "bin":
                    # Optional debug canvas (copy of vis) to draw skin thumb
                    skin_dbg = vis if args.skin_debug else None
                    auto_result = _auto_diameter_binary(
                        gray, band_rect, keep_mask, band_orient,
                        args.auto_contrast, args.auto_sharp, args.auto_gamma,
                        full_bgr=frame, use_skin=args.skin, skin_debug_draw=skin_dbg
                    )
                    if auto_result is None:
                        # graceful fallback to edge method if binary+skin fails
                        auto_result = _auto_diameter_px(gray, band_rect, keep_mask, band_orient)
                else:
                    auto_result = _auto_diameter_px(gray, band_rect, keep_mask, band_orient)

                # Draw if we got a result (for either method)
                if auto_result:
                    p1, p2, diameter_px = auto_result
                    mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
                    mpp = local_scale_m_per_px(Hinv, mid[0], mid[1]) if Hinv is not None else gscale_smoothed
                    if mpp:
                        mm = diameter_px * mpp * 1000.0
                        circ_mm = np.pi * mm
                        cv2.line(vis, p1, p2, (0, 180, 255), 2)
                        _put_text(vis, f"[auto] D={mm:.1f} mm  C={circ_mm:.1f} mm", (8, vis.shape[0]-32), (0, 180, 255))


            # Click fallback (uses local scale at midpoint if Hinv available)
            p1 = click_pts[0] if len(click_pts) >= 1 else None
            p2 = click_pts[1] if len(click_pts) >= 2 else None
            seg_mpp: Optional[float] = None
            if p1 and p2:
                mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
                seg_mpp = local_scale_m_per_px(Hinv, mid[0], mid[1]) if Hinv is not None else gscale_smoothed
            _draw_measurement_overlay(vis, p1, p2, seg_mpp)

            # GUI / headless
            if args.no_gui:
                if i % args.fps == 0:
                    msg = f"[{time.strftime('%H:%M:%S')}] Hinv={'Y' if Hinv is not None else 'N'} auto={'Y' if auto_on else 'N'} orient={band_orient}"
                    print(msg)
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
                if key == ord('o'):
                    band_orient = 'v' if band_orient == 'h' else 'h'
                # Move band: arrows move perpendicular center for horizontal, or parallel center for vertical too
                if key == 82:  # Up
                    if band_orient == 'h':
                        band_cy = max(band_size // 2 + 4, band_cy - 6)
                    else:
                        band_cx = max(band_size // 2 + 4, band_cx - 6)
                if key == 84:  # Down
                    if band_orient == 'h':
                        band_cy = min(h - band_size // 2 - 4, band_cy + 6)
                    else:
                        band_cx = min(w - band_size // 2 - 4, band_cx + 6)

            i += 1

    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
