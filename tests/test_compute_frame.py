import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from aberration_animation import (
    compute_frame,
    surf_y,
    ys,
    plane_x,
    radius,
    incoming_final_angles,
)


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
    for idx, (y, m, kink) in enumerate(zip(ys, slopes, kinks)):
        expected_x = np.sqrt(r**2 - y**2) - r + plane_x
        assert np.isclose(kink, expected_x)
        normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        in_angle = t * incoming_final_angles[idx]
        phi_in = abs(in_angle - normal_angle)
        out_angle = abs(normal_angle - np.arctan(m))
        if np.sin(phi_in) == 0 and out_angle == 0:
            continue
        ratio = np.sin(phi_in) / np.sin(out_angle)
        assert np.isclose(ratio, 1 / n_ratio)


def test_surface_at_plane():
    t = 1.0
    x, slopes, _, _ = compute_frame(t)
    # at t=1 the surface should be nearly a plane located at ``plane_x``
    assert np.allclose(x, plane_x, atol=5e-3)

    # expected outgoing slopes from Snell's law with a flat surface
    in_angles = incoming_final_angles
    r = 0.6 / np.sin(np.arcsin(0.6 / 50))  # r at t=1
    expected_slopes = []
    for angle_in, y in zip(in_angles, ys):
        normal_angle = np.arctan2(y, np.sqrt(r**2 - y**2))
        phi_in = abs(angle_in - normal_angle)
        phi_out = np.arcsin(
            np.clip(np.sin(phi_in) * 1.4, -1 + 1e-9, 1 - 1e-9)
        )
        orient = normal_angle + np.sign(angle_in - normal_angle) * phi_out
        expected_slopes.append(np.tan(orient))
    assert np.allclose(slopes, expected_slopes)


def test_symmetry_of_rays():
    t = 0.6
    x, slopes, intercepts, kinks = compute_frame(t)
    # surface is symmetric so ray parameters must be symmetric as well
    for i in range(len(ys) // 2):
        j = -(i + 1)
        assert np.isclose(slopes[i], -slopes[j])
        assert np.isclose(intercepts[i], -intercepts[j])
        assert np.isclose(kinks[i], kinks[j])


def test_no_refraction_when_n_equal():
    t = 0.4
    x, slopes, intercepts, kinks = compute_frame(t, n_ratio=1.0)
    in_angles = t * incoming_final_angles
    expected_slopes = np.tan(in_angles)
    assert np.allclose(slopes, expected_slopes)

    # Intercepts should be consistent with a line of slope ``expected_slopes``
    # passing through the surface point ``kinks`` at ``ys``
    expected_intercepts = ys - expected_slopes * kinks
    assert np.allclose(intercepts, expected_intercepts)
