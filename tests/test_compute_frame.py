import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aberration_animation import compute_frame, surf_y, scenario1, scenario2


def test_compute_frame_shapes():
    x, slopes, intercepts = compute_frame(0.5)
    assert x.shape == surf_y.shape
    assert slopes.shape[0] == scenario1.shape[0]
    assert intercepts.shape[0] == scenario1.shape[0]


def test_compute_frame_endpoints():
    x0, s0, b0 = compute_frame(0.0)
    x1, s1, b1 = compute_frame(1.0)

    expected_s0 = []
    expected_b0 = []
    for row in scenario1:
        p1 = row[1]
        p2 = row[2]
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
        expected_s0.append(m)
        expected_b0.append(b)
    assert np.allclose(s0, expected_s0)
    assert np.allclose(b0, expected_b0)

    expected_s1 = []
    expected_b1 = []
    for row in scenario2:
        p1 = row[1]
        p2 = row[2]
        m = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b = p1[1] - m * p1[0]
        expected_s1.append(m)
        expected_b1.append(b)
    assert np.allclose(s1, expected_s1)
    assert np.allclose(b1, expected_b1)

