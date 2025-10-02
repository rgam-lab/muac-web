#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2


def list_cameras(max_index: int = 10) -> List[int]:
    """Best-effort probe for available camera indices."""
    found = []
    for i in range(max_index + 1):
        cap = cv2.VideoCapture(i)
        ok = cap.isOpened()
        cap.release()
        if ok:
            found.append(i)
    return found


def open_camera(index: int = 0, width: int = 1280, height: int = 720, fps: int = 30) -> Tuple[cv2.VideoCapture, Optional[Tuple[int,int,int]]]:
    """Open a camera with basic property hints. Returns (cap, fourcc_tuple_or_None)."""
    cap = cv2.VideoCapture(index)
    # Hints (may not be honored by all backends)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    # Try to read one frame to validate
    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"Failed to open camera index {index}")
    # Try to get FOURCC (if available)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC)) if cap.get(cv2.CAP_PROP_FOURCC) else 0
    f = (
        (fourcc & 0xFF),
        ((fourcc >> 8) & 0xFF),
        ((fourcc >> 16) & 0xFF),
        ((fourcc >> 24) & 0xFF),
    )
    return cap, f if fourcc != 0 else None
