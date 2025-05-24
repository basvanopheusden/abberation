import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from aberration_animation import compute_snell_paths


def test_angle_changes_only_at_surface():
    rays = compute_snell_paths(5)
    # Each ray should consist of start, surface intersection, and final point
    assert rays.shape[1] == 3


def test_snell_law_constant():
    n1 = 1.0
    n2 = 1.5
    rays = compute_snell_paths(7, n1=n1, n2=n2)
    ratios = []
    for start, plane, final in rays:
        theta_in = np.arctan2(plane[1] - start[1], plane[0] - start[0])
        theta_out = np.arctan2(final[1] - plane[1], final[0] - plane[0])
        if np.isclose(theta_in, 0.0) and np.isclose(theta_out, 0.0):
            continue
        ratios.append(abs(np.sin(theta_in) / np.sin(theta_out)))
    assert ratios, "No ratios computed"
    # All ratios should be equal and match the refractive index ratio
    assert np.allclose(ratios, ratios[0])
    assert np.isclose(ratios[0], n2 / n1)
