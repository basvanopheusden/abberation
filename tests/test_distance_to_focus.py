import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aberration.analysis import distance_to_focus
from aberration.optics import compute_frame


def test_distance_scalar():
    result = distance_to_focus(1.0, 0.0)
    expected = np.abs((0.0 - 0.0) / 1.0 - 1.2)
    assert np.isclose(result, expected)


def test_distance_vectorized():
    _, slopes, intercepts, _ = compute_frame(0.2)
    result = distance_to_focus(slopes, intercepts)
    with np.errstate(divide="ignore", invalid="ignore"):
        expected = np.abs(np.where(slopes != 0, (0.0 - intercepts) / slopes, 1.2) - 1.2)
    assert np.allclose(result, expected)


def test_custom_focal_point():
    result = distance_to_focus(1.0, 0.0, focal_point=(2.0, 0.1))
    expected = np.abs((0.1 - 0.0) / 1.0 - 2.0)
    assert np.isclose(result, expected)

