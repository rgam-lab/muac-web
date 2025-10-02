#!/usr/bin/env python3
"""
Environment sanity checker for ArUco Toolkit.
Prints Python/OpenCV versions, ArUco availability, and a short build info line.
"""

from __future__ import annotations

import platform
import sys


def main() -> int:
    py_ver = platform.python_version()
    print(f"Python version: {py_ver}")

    try:
        import cv2  # type: ignore

        print(f"OpenCV version: {cv2.__version__}")
        has_aruco = hasattr(cv2, "aruco")
        print(f"ArUco available: {has_aruco}")

        # Short build info line
        try:
            bi = cv2.getBuildInformation()
            # Grab the first line and any GUI backends hint
            first_line = bi.splitlines()[0].strip()
            gui_flags = []
            for key in ("GTK", "Qt", "Cocoa", "Win32UI", "OpenGL", "FFmpeg"):
                if key in bi:
                    gui_flags.append(key)
            print("Build info:", first_line)
            if gui_flags:
                print("Detected components:", ", ".join(sorted(set(gui_flags))))
        except Exception as e:
            print(f"Build info: <unavailable> ({e})")

        # Minimal functional probe if aruco present
        if has_aruco:
            try:
                # Try a small, common dictionary
                _ = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                print("ArUco dictionary probe: OK")
            except Exception as e:
                print(f"ArUco dictionary probe: FAILED ({e})")
                return 2

        # Exit codes: 0 OK, 1 missing cv2, 2 aruco problems
        return 0

    except Exception as e:
        print(f"Error importing OpenCV: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
