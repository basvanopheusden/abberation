import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import aberration_animation as aa


def test_rays_change_only_at_surface():
    plane_x = aa.plane_x
    for ray in aa.scenario2:
        # all rays have 3 points: start, surface, final
        assert len(ray) == 3
        # the bend happens at the interface x coordinate
        assert np.allclose(ray[1, 0], plane_x)


def test_snells_law_constant():
    plane_x = aa.plane_x
    final_x = aa.final_x
    k = aa.n_ratio
    ratios = []
    for start, surface, final in aa.scenario2:
        slope_in = (surface[1] - start[1]) / (surface[0] - start[0])
        slope_out = (final[1] - surface[1]) / (final[0] - surface[0])
        sin_in = abs(slope_in) / np.sqrt(1 + slope_in ** 2)
        sin_out = abs(slope_out) / np.sqrt(1 + slope_out ** 2)
        if sin_out == 0 and sin_in == 0:
            continue
        ratios.append(sin_in / sin_out)
    ratios = np.array(ratios)
    assert np.allclose(ratios, k)
