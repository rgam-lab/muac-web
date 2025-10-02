#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def build_gamma_lut(gamma: float) -> np.ndarray:
    # x^gamma (gamma<1 brightens, >1 darkens) on [0..255]
    gamma = max(0.1, min(5.0, float(gamma)))
    lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.float32)
    return np.clip(lut, 0, 255).astype(np.uint8)


def unsharp_bgr(img: np.ndarray, amount: float, sigma: float = 1.2) -> np.ndarray:
    """
    Unsharp mask on BGR image.
    amount ≈ 0..2.0 (0=off). Higher increases crispness but can halo.
    """
    if amount <= 1e-6:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # sharpened = img*(1+amount) - blur*amount
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def adjust_saturation_bgr(img: np.ndarray, sat_scale: float) -> np.ndarray:
    """
    Scale saturation in HSV by sat_scale (0..3).
    """
    if abs(sat_scale - 1.0) < 1e-3:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip((s.astype(np.float32) * sat_scale), 0, 255).astype(np.uint8)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_enhance(img_bgr: np.ndarray,
                  alpha: float, beta: float,
                  gamma: float,
                  sat_scale: float,
                  sharp_amount: float,
                  clahe_clip: float | None) -> tuple[np.ndarray, np.ndarray]:
    """
    Enhancement pipeline:
      1) Brightness/Contrast in BGR (convertScaleAbs with alpha, beta)
      2) Gamma via LUT on BGR
      3) Saturation scale in HSV
      4) Unsharp on BGR
      5) Gray + optional CLAHE on Gray
    Returns: (enhanced_bgr, enhanced_gray)
    """
    # 1) Brightness/Contrast
    img = cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)

    # 2) Gamma LUT on BGR
    if abs(gamma - 1.0) > 1e-3:
        lut = build_gamma_lut(gamma)
        img = cv2.LUT(img, lut)

    # 3) Saturation
    img = adjust_saturation_bgr(img, sat_scale)

    # 4) Unsharp
    img = unsharp_bgr(img, sharp_amount)

    # 5) Gray + optional CLAHE
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if clahe_clip is not None and clahe_clip > 0.0:
        clahe = cv2.createCLAHE(clipLimit=float(clahe_clip), tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    return img, gray


def make_binary(gray: np.ndarray, use_otsu: bool, thr_val: int) -> np.ndarray:
    if use_otsu:
        # Try both polarities, pick the one with smaller (but non-trivial) foreground area
        _, bin1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, bin2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        f1 = (bin1 > 0).mean()
        f2 = (bin2 > 0).mean()
        # Prefer smaller non-extreme foreground
        if 0.02 < f2 < 0.98 and (f2 <= f1 or not (0.02 < f1 < 0.98)):
            return bin2
        return bin1
    # Manual threshold (binary, not inverted)
    _, b = cv2.threshold(gray, thr_val, 255, cv2.THRESH_BINARY)
    return b


def tile4(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, maxw: int = 1280) -> np.ndarray:
    """
    Make a 2x2 visualization grid. Each image gets resized to same height.
    """
    def prep(x):
        if x.ndim == 2:
            x = cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
        return x

    a, b, c, d = map(prep, (a, b, c, d))
    h = min(a.shape[0], b.shape[0], c.shape[0], d.shape[0])
    # Resize all to same size (half width of maxw for each tile)
    target_w = maxw // 2
    target_h = int(h * (target_w / max(a.shape[1], 1)))
    a = cv2.resize(a, (target_w, target_h))
    b = cv2.resize(b, (target_w, target_h))
    c = cv2.resize(c, (target_w, target_h))
    d = cv2.resize(d, (target_w, target_h))
    top = np.hstack([a, b])
    bot = np.hstack([c, d])
    grid = np.vstack([top, bot])
    return grid


def on_trackbar(_=None):
    # no-op; values read inside main loop
    pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Webcam enhancement tuner (brightness/contrast/gamma/saturation/sharpness/CLAHE/threshold)")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--no-gui", action="store_true", help="Run without windows (not super useful here)")
    ap.add_argument("--out-dir", type=Path, default=Path("out"), help="Where to save snapshots (space key)")
    args = ap.parse_args()

    if args.no_gui:
        print("This tool is designed for interactive GUI use. Remove --no-gui.")
        return 1

    cap = cv2.VideoCapture(args.camera_index)
    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps > 0:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    win = "Enhance Tuner (q/ESC quit, space pause, s save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    # Trackbars (ranges & defaults)
    # Contrast alpha: 0.1..3.0 (slider 10..300), default 100 → 1.0
    cv2.createTrackbar("Contrast % (alpha)", win, 100, 300, on_trackbar)  # 100 → 1.0
    # Brightness beta: -100..+100 (slider 0..200)
    cv2.createTrackbar("Brightness (beta)", win, 100, 200, on_trackbar)   # 100 → 0
    # Gamma: 0.1..3.0 (slider 10..300), default 100 → 1.0
    cv2.createTrackbar("Gamma x100", win, 100, 300, on_trackbar)
    # Saturation: 0..3.0 (slider 0..300), default 100 → 1.0
    cv2.createTrackbar("Saturation %", win, 100, 300, on_trackbar)
    # Sharpness: 0..200 → 0..2.0
    cv2.createTrackbar("Sharpness %", win, 0, 200, on_trackbar)
    # CLAHE clip: 0..300 → 0..3.0 (0=off)
    cv2.createTrackbar("CLAHE clip x100", win, 0, 300, on_trackbar)
    # Threshold (manual)
    cv2.createTrackbar("Thresh (manual)", win, 128, 255, on_trackbar)
    # Otsu toggle (0/1)
    cv2.createTrackbar("Otsu (0/1)", win, 1, 1, on_trackbar)

    paused = False
    last_save = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed.")
                break
        frame_disp = frame.copy()

        # Read trackbars → actual params
        contrast_pct = cv2.getTrackbarPos("Contrast % (alpha)", win)      # 0..300
        alpha = max(0.1, contrast_pct / 100.0 * 1.0)                      # 0.1..3.0 (linear)
        beta_slider = cv2.getTrackbarPos("Brightness (beta)", win)        # 0..200
        beta = float(beta_slider - 100)                                   # -100..+100

        gamma_x100 = cv2.getTrackbarPos("Gamma x100", win)                # 10..300 typical
        gamma = max(0.1, gamma_x100 / 100.0)                              # 0.1..3.0

        sat_pct = cv2.getTrackbarPos("Saturation %", win)                 # 0..300
        sat_scale = max(0.0, sat_pct / 100.0)                             # 0..3.0

        sharp_pct = cv2.getTrackbarPos("Sharpness %", win)                # 0..200
        sharp_amount = max(0.0, min(2.0, sharp_pct / 100.0))              # 0..2.0

        clahe_x100 = cv2.getTrackbarPos("CLAHE clip x100", win)           # 0..300
        clahe_clip = (clahe_x100 / 100.0) if clahe_x100 > 0 else None     # None → off

        thr_val = cv2.getTrackbarPos("Thresh (manual)", win)
        use_otsu = bool(cv2.getTrackbarPos("Otsu (0/1)", win))

        # Apply enhancements
        enh_bgr, enh_gray = apply_enhance(frame_disp, alpha, beta, gamma, sat_scale, sharp_amount, clahe_clip)
        # Binary
        binary = make_binary(enh_gray, use_otsu, thr_val)

        # Compose 2x2 grid
        grid = tile4(frame_disp, enh_bgr, enh_gray, binary, maxw=1280)

        # Annotate params
        overlay = grid.copy()
        text = f"alpha={alpha:.2f}  beta={beta:.0f}  gamma={gamma:.2f}  sat={sat_scale:.2f}  sharp={sharp_amount:.2f}  clahe={'off' if clahe_clip is None else f'{clahe_clip:.2f}'}  thr={'OTSU' if use_otsu else thr_val}"
        cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 24), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, grid, 0.4, 0, grid)
        cv2.putText(grid, text, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 255, 240), 1, cv2.LINE_AA)

        cv2.imshow(win, grid)
        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord('q')):
            break
        if key == ord(' '):
            paused = not paused
        if key == ord('s'):
            ts = int(time.time())
            out1 = args.out_dir / f"tuner_{ts}_enh.png"
            out2 = args.out_dir / f"tuner_{ts}_bin.png"
            cv2.imwrite(str(out1), enh_bgr)
            cv2.imwrite(str(out2), binary)
            print(f"Saved {out1} and {out2}")

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
