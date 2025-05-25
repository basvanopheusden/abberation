import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from aberration.utils import build_patch, init_axes
from aberration import params


def test_build_patch_shapes():
    x = np.linspace(-1.0, 1.0, params.surf_y.size)
    left, right = build_patch(x)
    assert left.ndim == 2 and right.ndim == 2
    assert left.shape[1] == 2
    assert right.shape[1] == 2


def test_init_axes_returns_objects(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    fig, ax, left_patch, right_patch = init_axes()
    assert fig is not None
    assert ax.get_xlim() == params.xlim
    assert left_patch.get_xy().shape[1] == 2
    plt.close(fig)
