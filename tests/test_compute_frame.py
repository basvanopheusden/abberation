import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aberration_animation import compute_frame, surf_y, ys, plane_x, radius


def test_compute_frame_shapes():
    x, slopes, intercepts, kinks = compute_frame(0.5)
    assert x.shape == surf_y.shape
    assert slopes.shape[0] == ys.shape[0]
    assert intercepts.shape[0] == ys.shape[0]
    assert kinks.shape[0] == ys.shape[0]


def test_snells_law_and_kinks():
    t = 0.3
    n_ratio = 1.4
    x, slopes, intercepts, kinks = compute_frame(t, n_ratio=n_ratio)
    start_angle = np.arcsin(np.clip(0.6 / radius, -1 + 1e-9, 1 - 1e-9))
    end_angle = np.arcsin(0.6 / 50)
    angle = (1 - t) * start_angle + t * end_angle
    r = 0.6 / np.sin(angle)
    for y, m, kink in zip(ys, slopes, kinks):
        expected_x = np.sqrt(r**2 - y**2) - r + plane_x
        assert np.isclose(kink, expected_x)
        normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        out_angle = abs(normal_angle - np.arctan(m))
        if np.sin(abs(normal_angle)) == 0 and out_angle == 0:
            continue
        ratio = np.sin(abs(normal_angle)) / np.sin(out_angle)
        assert np.isclose(ratio, 1 / n_ratio)
