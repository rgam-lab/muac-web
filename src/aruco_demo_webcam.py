#!/usr/bin/env python3
"""
Live ArUco demo (webcam).

Examples:
    # GUI mode (default)
    python src/aruco_demo_webcam.py --camera-index 0 --dict 4X4_50 --marker-length 0.04

    # Headless logging + periodic frame saves
    python src/aruco_demo_webcam.py --no-gui --save-dir out/ --fps 30
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from utils.aruco_helpers import get_dictionary, create_detector_params
from utils.camera import open_camera, list_cameras


def _detect(gray: np.ndarray, dict_name: str):
    dictionary = get_dictionary(dict_name)
    params = create_detector_params()
    if hasattr(cv2.aruco, "ArucoDetector"):
        det = cv2.aruco.ArucoDetector(dictionary, params)
        return det.detectMarkers(gray)
    else:
        return cv2.aruco.detectMarkers(gray, dictionary, parameters=params)


def main() -> int:
    ap = argparse.ArgumentParser(description="Webcam ArUco demo with GUI or headless modes.")
    ap.add_argument("--camera-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--dict", default="4X4_50")
    ap.add_argument("--marker-length", type=float, default=0.04, help="meters; only used if pose is added later")
    ap.add_argument("--show-axes", action="store_true", help="(reserved) pose axes when intrinsics are added")
    ap.add_argument("--no-gui", action="store_true")
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--record", type=Path, default=None, help="Optional output mp4 path (best-effort)")
    args = ap.parse_args()

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)

    # Camera open
    cap, fourcc = open_camera(args.camera_index, args.width, args.height, args.fps)
    if fourcc:
        print(f"Camera FOURCC: {''.join(map(chr, fourcc))}")

    # Optional recorder (best-effort; may fail depending on codecs)
    writer: Optional[cv2.VideoWriter] = None
    if args.record:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.record), fourcc, float(args.fps), (args.width, args.height))
        if not writer.isOpened():
            print("Warning: failed to open VideoWriter; recording disabled.")
            writer = None

    # Main loop
    i = 0
    last_saved = time.time()
    try:
        win = "ArUco Webcam" if not args.no_gui else None
        if win:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)

        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, rejected = _detect(gray, args.dict)

            # Draw detections
            if ids is not None and len(ids) > 0:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                # (Pose axes reserved for later when intrinsics exist)
                count = int(len(ids))
            else:
                count = 0

            if writer:
                writer.write(frame)

            if args.no_gui:
                # Periodic logging + save every ~1s if requested
                if i % args.fps == 0:
                    print(f"[{time.strftime('%H:%M:%S')}] detected={count}")
                if args.save_dir and time.time() - last_saved >= 1.0:
                    out_path = args.save_dir / f"frame_{int(time.time())}.png"
                    cv2.imwrite(str(out_path), frame)
                    last_saved = time.time()
            else:
                cv2.imshow(win, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')):  # ESC or 'q'
                    break

            i += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
