import importlib
from pathlib import Path


def test_cv2_and_aruco_present():
    cv2 = importlib.import_module("cv2")
    assert hasattr(cv2, "__version__")
    assert hasattr(cv2, "aruco"), "cv2.aruco module is missing (need opencv-contrib-python)"

def test_helpers_dictionary_and_marker(tmp_path: Path):
    """Smoke test: get dictionary and generate one marker to disk."""
    cv2 = importlib.import_module("cv2")
    helpers = importlib.import_module("src.utils.aruco_helpers")
    d = helpers.get_dictionary("4X4_50")
    assert d is not None

    out = tmp_path / "markers"
    paths = helpers.print_markers("4X4_50", [0], size_px=100, out_dir=out, border_px=10)
    assert len(paths) == 1
    assert paths[0].exists()
