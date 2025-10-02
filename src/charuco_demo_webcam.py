#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from utils.camera import open_camera
from utils.charuco_helpers import (
    CharucoConfig,
    detect_charuco,
    draw_charuco,
    estimate_homography_from_charuco,
    estimate_scale_m_per_px_from_charuco,  # fallback
    local_scale_m_per_px,
    project_board_polygon_px,
)

Point = Tuple[int, int]


def _put_text(img, text, org, color=(255, 255, 255), scale=0.6, thickness=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_measurement_overlay(frame: np.ndarray, p1: Optional[Point], p2: Optional[Point], m_per_px: Optional[float]) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, 28), (0, 0, 0), -1)
    _put_text(frame, "Click 2 points | 'c' clear | 's' save | 'q' quit | 'a' auto | 'o' orient | arrows move band | 'R' skin view", (8, 20))
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


# ====== Enhancement utilities (same semantics as tuner) ======

def _build_gamma_lut(gamma: float) -> np.ndarray:
    gamma = max(0.1, min(5.0, float(gamma)))
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    return np.clip(lut, 0, 255).astype(np.uint8)


def _unsharp_bgr(img: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    if amount <= 1e-6:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _adjust_saturation_bgr(img: np.ndarray, sat_scale: float) -> np.ndarray:
    if abs(sat_scale - 1.0) < 1e-3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip((s.astype(np.float32) * sat_scale), 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _apply_enhance_bgr_to_gray(
    band_bgr: np.ndarray,
    alpha: float, beta: float,
    gamma: float,
    sat_scale: float,
    sharp_amount: float,
    clahe_clip: Optional[float],
) -> np.ndarray:
    """
    Enhancement pipeline:
      1) Brightness/Contrast in BGR (alpha, beta)
      2) Gamma LUT on BGR (x^gamma)
      3) Saturation scale in HSV
      4) Unsharp on BGR
      5) Convert to Gray (+ optional CLAHE)
    Returns enhanced grayscale band.
    """
    # 1) Brightness/Contrast
    img = cv2.convertScaleAbs(band_bgr, alpha=alpha, beta=beta)
    # 2) Gamma
    if abs(gamma - 1.0) > 1e-3:
        lut = _build_gamma_lut(gamma)
        img = cv2.LUT(img, lut)
    # 3) Saturation
    img = _adjust_saturation_bgr(img, sat_scale)
    # 4) Unsharp
    img = _unsharp_bgr(img, sharp_amount)
    # 5) Gray + CLAHE (optional)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe_clip is not None and clahe_clip > 0.0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        gray = clahe.apply(gray)
    return gray


# ====== Skin mask (broad but with highlight suppression) ======

def _skin_mask_rgb(img_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    m_ycrcb = ((Cr >= 120) & (Cr <= 180) & (Cb >= 70) & (Cb <= 150))

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hc, Sc, Vc = cv2.split(hsv)
    m_hsv = ((Hc <= 50) & (Sc >= 15) & (Vc >= 15))

    m = (m_ycrcb | m_hsv).astype(np.uint8)
    highlights = ((Vc >= 225) & (Sc <= 25))
    very_bright = (Vc >= 245)
    m[highlights | very_bright] = 0

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m  # {0,1}


# ====== Build a band mask (manual/otsu threshold supported) ======

def _make_band_mask(
    frame_bgr: np.ndarray,
    gray_full: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask_full: np.ndarray,
    use_skin: bool,
    # enhancement params (from CLI)
    bc_alpha: float, bc_beta: float, gamma: float, sat_scale: float, sharp_amount: float, clahe_clip: Optional[float],
    # threshold params (from CLI)
    use_otsu: bool, manual_thr: int,
    skin_debug_draw: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Build a clean binary mask of the foreground inside the band:
      - Optional broad skin filter
      - Enhancement in BGR → gray
      - Manual threshold (default) or Otsu
      - Remove board region
      - Morph cleanup
    Returns binary {0,1} in band coords.
    """
    x0, y0, x1, y1 = band_rect
    band_bgr = frame_bgr[y0:y1, x0:x1]
    km = keep_mask_full[y0:y1, x0:x1]  # 1=keep, 0=remove (board)

    # Optional skin prefilter
    pre_mask = np.ones((band_bgr.shape[0], band_bgr.shape[1]), dtype=np.uint8)
    if use_skin:
        sm = _skin_mask_rgb(band_bgr)  # {0,1}
        pre_mask = sm
        if skin_debug_draw is not None:
            thumb = (sm * 255)
            th = min(80, thumb.shape[0])
            tw = min(80, thumb.shape[1])
            thumb = cv2.resize(thumb, (tw, th), interpolation=cv2.INTER_NEAREST)
            skin_debug_draw[4:4+th, -tw-4:-4] = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)

    # Enhancement → gray
    band_gray = _apply_enhance_bgr_to_gray(
        band_bgr,
        alpha=bc_alpha, beta=bc_beta,
        gamma=gamma, sat_scale=sat_scale,
        sharp_amount=sharp_amount, clahe_clip=clahe_clip
    )

    # Threshold
    if use_otsu:
        # try both polarities and pick compact foreground
        _, b1 = cv2.threshold(band_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, b2 = cv2.threshold(band_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        f1 = (b1 > 0).mean()
        f2 = (b2 > 0).mean()
        chosen = b2 if 0.02 < f2 < 0.98 and (f2 <= f1 or not (0.02 < f1 < 0.98)) else b1
    else:
        _, chosen = cv2.threshold(band_gray, int(manual_thr), 255, cv2.THRESH_BINARY)

    mask = (chosen > 0).astype(np.uint8)
    mask[km == 0] = 0
    mask &= pre_mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask * 255, cv2.MORPH_CLOSE, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    return (mask > 0).astype(np.uint8)


# ====== Ellipse-fit auto (uses the band mask above) ======

def _auto_diameter_ellipse(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask_full: np.ndarray,
    orient: str,
    use_skin: bool,
    # enhancement & threshold params
    bc_alpha: float, bc_beta: float, gamma: float, sat_scale: float, sharp_amount: float, clahe_clip: Optional[float],
    use_otsu: bool, manual_thr: int,
) -> Optional[Tuple[Point, Point, float]]:
    x0, y0, x1, y1 = band_rect
    h_band = max(1, y1 - y0)
    w_band = max(1, x1 - x0)

    mask = _make_band_mask(
        frame_bgr, gray, band_rect, keep_mask_full, use_skin,
        bc_alpha, bc_beta, gamma, sat_scale, sharp_amount, clahe_clip,
        use_otsu, manual_thr
    )
    if mask.sum() < 20:
        return None

    num_labels, labels = cv2.connectedComponents(mask, connectivity=8)
    if num_labels <= 1:
        return None
    areas = [(labels == i).sum() for i in range(1, num_labels)]
    best_idx = int(np.argmax(areas)) + 1
    best_area = areas[best_idx - 1]
    if best_area < 0.003 * (h_band * w_band):
        return None
    comp = (labels == best_idx).astype(np.uint8) * 255

    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if len(cnt) < 20:
        return None

    try:
        (cx, cy), (ax_w, ax_h), angle_deg = cv2.fitEllipse(cnt)
    except cv2.error:
        return None

    minor_len = float(min(ax_w, ax_h))
    theta = np.deg2rad(angle_deg)
    v_minor = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    p1_band = np.array([cx, cy], dtype=np.float64) - 0.5 * minor_len * v_minor
    p2_band = np.array([cx, cy], dtype=np.float64) + 0.5 * minor_len * v_minor

    p1 = (int(round(x0 + p1_band[0])), int(round(y0 + p1_band[1])))
    p2 = (int(round(x0 + p2_band[0])), int(round(y0 + p2_band[1])))

    if not (x0 <= p1[0] < x1 and y0 <= p1[1] < y1 and x0 <= p2[0] < x1 and y0 <= p2[1] < y1):
        return None

    return p1, p2, minor_len


# ====== Binary projection auto (also uses the band mask) ======

def _auto_diameter_binary(
    frame_bgr: np.ndarray,
    gray: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask: np.ndarray,
    orient: str,
    use_skin: bool,
    # enhancement & threshold params
    bc_alpha: float, bc_beta: float, gamma: float, sat_scale: float, sharp_amount: float, clahe_clip: Optional[float],
    use_otsu: bool, manual_thr: int,
) -> Optional[Tuple[Point, Point, float]]:
    x0, y0, x1, y1 = band_rect
    mask = _make_band_mask(
        frame_bgr, gray, band_rect, keep_mask, use_skin,
        bc_alpha, bc_beta, gamma, sat_scale, sharp_amount, clahe_clip,
        use_otsu, manual_thr
    )
    if mask.sum() < 20:
        return None

    if orient == 'h':
        profile = mask.mean(axis=1).astype(np.float32)
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

    profile = mask.mean(axis=0).astype(np.float32)
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


# ====== Edge fallback (unchanged) ======

def _auto_diameter_px(
    gray: np.ndarray,
    band_rect: Tuple[int, int, int, int],
    keep_mask: np.ndarray,
    orient: str,
) -> Optional[Tuple[Point, Point, float]]:
    x0, y0, x1, y1 = band_rect
    band = gray[y0:y1, x0:x1]
    if band.size == 0:
        return None

    km = keep_mask[y0:y1, x0:x1]
    blur = cv2.GaussianBlur(band, (5, 5), 0)

    if orient == 'h':
        edges = cv2.Canny(blur, 40, 120)
        edges[km == 0] = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9))
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        profile = edges.mean(axis=1)
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
        return (xC, y0 + yT), (xC, y0 + yB), float(height)

    edges = cv2.Canny(blur, 40, 120)
    edges[km == 0] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    profile = edges.mean(axis=0)
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
    return (x0 + xL, yC), (x0 + xR, yC), float(width)


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
    # Auto method
    ap.add_argument("--auto-method", choices=["ellipse", "bin", "edge"], default="ellipse",
                    help="Auto method: 'ellipse' (robust), 'bin' (binary projection), or 'edge'")
    # Skin
    ap.add_argument("--skin", action="store_true", help="Enable broad skin prefilter in band (YCbCr+HSV union)")
    ap.add_argument("--skin-debug", action="store_true", help="Show a tiny skin mask indicator in the overlay")
    # === New: Enhancement & threshold defaults (your chosen values) ===
    ap.add_argument("--bc-alpha", type=float, default=3.00, help="Contrast gain (alpha), e.g., 3.0")
    ap.add_argument("--bc-beta", type=float, default=-100.0, help="Brightness offset (beta), e.g., -100")
    ap.add_argument("--gamma", type=float, default=3.00, help="Gamma exponent (x^gamma), e.g., 3.0")
    ap.add_argument("--sat", type=float, default=0.0, help="Saturation scale (0..3). 0 disables color; 1 keeps original")
    ap.add_argument("--sharp", type=float, default=0.0, help="Unsharp amount (0..2)")
    ap.add_argument("--clahe", type=float, default=0.0, help="CLAHE clip (0=off)")
    ap.add_argument("--use-otsu", action="store_true", help="Use Otsu instead of manual threshold")
    ap.add_argument("--thr", type=int, default=246, help="Manual threshold (0..255). Used when --use-otsu is NOT set")

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
    show_skin_full = False

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
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            det = detect_charuco(gray, cfg)

            skin_full = None
            if args.skin and show_skin_full:
                sm = _skin_mask_rgb(frame)  # full-frame mask
                skin_full = (sm * 255).astype(np.uint8)
                skin_full = cv2.cvtColor(skin_full, cv2.COLOR_GRAY2BGR)

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
            cv2.rectangle(vis, (0, 0), (vis.shape[1], 72), (0, 0, 0), -1)
            if Hinv is not None:
                _put_text(vis, f"pose-aware scale ✓ (pts={H_used})", (8, 20), (0, 255, 0))
            else:
                _put_text(vis, "pose-aware scale × (need more ChArUco)", (8, 20), (0, 255, 255))
            if gscale_smoothed:
                _put_text(vis, f"global ~ {gscale_smoothed*1000:.3f} mm/px", (8, 40), (200, 200, 255))

            # Show current enhancement / threshold params
            hdr = f"auto={ 'ON' if auto_on else 'OFF' }  method={args.auto_method}  skin={args.skin}  α={args.bc_alpha:.2f} β={args.bc_beta:.0f} γ={args.gamma:.2f} sat={args.sat:.2f} sharp={args.sharp:.2f} clahe={args.clahe:.2f} thr={'OTSU' if args.use_otsu else args.thr}"
            _put_text(vis, hdr, (8, 60), (180, 220, 255))

            # Guide band
            band_rect = _rect_from_center_band(h, w, band_cy, band_cx, band_size, band_orient)
            _draw_guide_band(vis, band_rect, band_orient)

            # Auto detection in band
            auto_result = None
            if auto_on:
                if args.auto_method == "ellipse":
                    auto_result = _auto_diameter_ellipse(
                        frame, gray, band_rect, keep_mask, band_orient, args.skin,
                        args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                        args.use_otsu, args.thr
                    )
                    if auto_result is None:
                        auto_result = _auto_diameter_binary(
                            frame, gray, band_rect, keep_mask, band_orient, args.skin,
                            args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                            args.use_otsu, args.thr
                        )
                        if auto_result is None:
                            auto_result = _auto_diameter_px(gray, band_rect, keep_mask, band_orient)

                elif args.auto_method == "bin":
                    auto_result = _auto_diameter_binary(
                        frame, gray, band_rect, keep_mask, band_orient, args.skin,
                        args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                        args.use_otsu, args.thr
                    )
                    if auto_result is None:
                        auto_result = _auto_diameter_ellipse(
                            frame, gray, band_rect, keep_mask, band_orient, args.skin,
                            args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                            args.use_otsu, args.thr
                        )
                    if auto_result is None:
                        auto_result = _auto_diameter_px(gray, band_rect, keep_mask, band_orient)

                else:  # edge
                    auto_result = _auto_diameter_px(gray, band_rect, keep_mask, band_orient)
                    if auto_result is None:
                        auto_result = _auto_diameter_ellipse(
                            frame, gray, band_rect, keep_mask, band_orient, args.skin,
                            args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                            args.use_otsu, args.thr
                        )
                    if auto_result is None:
                        auto_result = _auto_diameter_binary(
                            frame, gray, band_rect, keep_mask, band_orient, args.skin,
                            args.bc_alpha, args.bc_beta, args.gamma, args.sat, args.sharp, (args.clahe if args.clahe > 0 else None),
                            args.use_otsu, args.thr
                        )

                if auto_result:
                    p1, p2, diameter_px = auto_result
                    mid = (int((p1[0] + p2[0]) * 0.5), int((p1[1] + p2[1]) * 0.5))
                    mpp = local_scale_m_per_px(Hinv, mid[0], mid[1]) if Hinv is not None else gscale_smoothed
                    if mpp:
                        mm = diameter_px * mpp * 1000.0
                        circ_mm = np.pi * mm
                        cv2.line(vis, p1, p2, (0, 180, 255), 2)
                        _put_text(vis, f"[auto] D={mm:.1f} mm  C={circ_mm:.1f} mm", (8, vis.shape[0]-32), (0, 180, 255))

            # Click fallback
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
                if args.skin and show_skin_full:
                    sm = _skin_mask_rgb(frame)
                    skin_full = cv2.cvtColor((sm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                    cv2.imshow(win, skin_full)
                else:
                    cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('r'), ord('R')):
                    show_skin_full = not show_skin_full
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
                # Move band with arrows
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
