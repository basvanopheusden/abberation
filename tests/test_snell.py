import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import aberration_animation as aa


def test_single_angle_change():
    plane_x = aa.plane_x
    for ray in aa.scenario2:
        start, mid, end = ray
        # angle change occurs at the surface
        assert start[0] < plane_x < end[0]
        assert np.isclose(mid[0], plane_x)
        slope1 = (mid[1] - start[1]) / (mid[0] - start[0])
        slope2 = (end[1] - mid[1]) / (end[0] - mid[0])
        if not np.isclose(start[1], 0):
            assert not np.isclose(slope1, slope2)
        else:
            # central ray passes straight through
            assert np.isclose(slope1, 0)
            assert np.isclose(slope2, 0)


def test_snells_law_constant_ratio():
    plane_x = aa.plane_x
    ratios = []
    for ray in aa.scenario2:
        start, mid, end = ray
        slope1 = (mid[1] - start[1]) / (mid[0] - start[0])
        slope2 = (end[1] - mid[1]) / (end[0] - mid[0])
        theta1 = np.arctan(slope1)
        theta2 = np.arctan(slope2)
        if np.isclose(np.sin(theta2), 0):
            continue
        ratios.append(np.sin(theta1) / np.sin(theta2))
    assert ratios, "no valid rays for ratio"
    ratio_first = ratios[0]
    for r in ratios:
        assert np.isclose(r, ratio_first)
    assert np.isclose(ratio_first, aa.n2 / aa.n1)
