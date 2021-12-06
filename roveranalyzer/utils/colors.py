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
