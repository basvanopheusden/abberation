import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aberration_animation import distance_to_focus, compute_frame


def test_distance_scalar():
    result = distance_to_focus(1.0, 0.0)
    expected = np.abs(1.0 * 1.2 - 0.0) / np.sqrt(1.0 ** 2 + 1)
    assert np.isclose(result, expected)


def test_distance_vectorized():
    _, slopes, intercepts, _ = compute_frame(0.2)
    result = distance_to_focus(slopes, intercepts)
    expected = np.abs(slopes * 1.2 + intercepts) / np.sqrt(slopes ** 2 + 1)
    assert np.allclose(result, expected)
