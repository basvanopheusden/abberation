import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.animation import FuncAnimation
from plane_wave_animation import (
    _compute_focal_point,
    _line_circle_intersection,
    _incoming_angle,
    Animator,
)
from aberration import params


def test_compute_focal_point_common_intersection():
    slopes = np.array([-2.0, -1.0, 1.0, 2.0])
    intercepts = -slopes
    result = _compute_focal_point(slopes, intercepts)
    assert np.isclose(result, 1.0)


def test_line_circle_intersection_basic():
    x, y = _line_circle_intersection(0.0, 0.0, 1.0)
    assert np.isclose(x, params.plane_x)
    assert np.isclose(y, 0.0)


def test_line_circle_intersection_diagonal():
    x, y = _line_circle_intersection(1.0, 0.0, 1.0)
    expected = (0.4114378277661477, 0.4114378277661477)
    assert np.allclose((x, y), expected)


def test_incoming_angle_snells_law():
    result = _incoming_angle(0.0, 0.0)
    assert np.isclose(result, 0.0)

    out_angle = 0.4
    normal = 0.0
    expected = np.arcsin(np.sin(out_angle) / params.ref_index_ratio)
    result = _incoming_angle(out_angle, normal)
    assert np.isclose(result, expected)


def test_animator_run_and_update(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    anim = Animator()
    animation = anim.run()
    assert isinstance(animation, FuncAnimation)
    artists = anim._update(0)
    assert len(artists) == params.n_rays + 2
    for line in anim.lines:
        assert len(line.get_xdata()) == 3
