from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


class Colors:
    bright_blue = (13 / 255, 0 / 255, 107 / 255)  # blue HSV(244, 255, 207)
    bright_red = (207 / 255, 0 / 255, 0 / 255)  # red HSV(0, 255, 207)


_bwr_data = [
    (13 / 255, 0 / 255, 107 / 255),  # blue negative HSV(244, 255, 207)
    (1.0, 1.0, 1.0),  # white center
    (207 / 255, 0 / 255, 0 / 255),  # red positive HSV(0, 255, 207)
]
_cmap_bwr = LinearSegmentedColormap.from_list("crownet_bwr", _bwr_data)

_registerd = False


def _register_cmap():
    if not _register_cmap:
        cm.register_cmap("crownet_bwr", _cmap_bwr)
        _registerd = True


def white_cmap(n=256):
    map = np.array([(1.0, 1.0, 1.0, 1.0) for i in np.arange(n)])
    return ListedColormap(map)


def mono_cmap(
    replace_with=(0.0, 0.0, 0.0, 0.0),
    replace_index=(0, 1),
    base_color=0,
    cspace=(0.0, 1.0),
    n=256,
):
    start, stop = replace_index
    map = np.array([(0.0, 0.0, 0.0, 1.0) for i in np.arange(n)])
    map[:, base_color] = np.linspace(cspace[0], cspace[1], n)
    map[start:stop] = replace_with
    return ListedColormap(map)


def t_cmap(
    cmap_name,
    replace_index=(0, 1, 1.0),
    use_colors=(0, 256),
    zero_color=None,
):
    cmap = plt.get_cmap(name=cmap_name)
    start, stop, alpha = replace_index
    colors = np.array(cmap(np.linspace(0, cmap.N, dtype=int)))
    colors = colors[use_colors[0] : use_colors[1]]
    colors[start:stop, -1] = alpha
    if zero_color is not None:
        colors[0] = zero_color
    return ListedColormap(colors)


class PlotOptions(Enum):
    COUNT = (1, "counts")
    DENSITY = (2, "density")
    DENSITY_SMOOTH = (3, "density_smooth")
