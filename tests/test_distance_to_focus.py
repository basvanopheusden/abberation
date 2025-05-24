import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aberration.analysis import distance_to_focus
from aberration.optics import compute_frame


def test_distance_scalar():
    result = distance_to_focus(1.0, 0.0)
    expected = np.abs(1.0 * 1.2 - 0.0) / np.sqrt(1.0 ** 2 + 1)
    assert np.isclose(result, expected)


def test_distance_vectorized():
    _, slopes, intercepts, _ = compute_frame(0.2)
    result = distance_to_focus(slopes, intercepts)
    expected = np.abs(slopes * 1.2 + intercepts) / np.sqrt(slopes ** 2 + 1)
    assert np.allclose(result, expected)

