import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aberration.optics import compute_frame
from aberration.analysis import (
    distance_to_focus,
    total_distance_to_focus,
    find_optimal_max_in_angle,
)
from aberration import params


def test_total_distance_matches_manual_sum():
    t = 0.5
    max_angle = 0.2
    in_angles = np.linspace(max_angle, -max_angle, params.n_rays)
    _, slopes, intercepts, _ = compute_frame(t, incoming_angles=in_angles)
    expected = np.sum(distance_to_focus(slopes, intercepts))
    result = total_distance_to_focus(t, max_angle)
    assert np.isclose(result, expected)


def test_find_optimal_matches_grid_search():
    t = 0.3
    search = np.linspace(0.1, 0.4, 7)
    best, dist = find_optimal_max_in_angle(t, search_angles=search)
    distances = [total_distance_to_focus(t, angle) for angle in search]
    idx = int(np.argmin(distances))
    assert np.isclose(best, search[idx])
    assert np.isclose(dist, distances[idx])

