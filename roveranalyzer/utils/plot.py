import itertools
import random

import matplotlib.pyplot as plt
import pandas as pd


class _PlotUtil:
    hatch_patterns = ("||", "--", "++", "x", "\\", "*", "|", "-", "+")
    plot_markers = ["o", "x", "*", ".", "v", "1", "2", "3", "4"]
    plot_colors = ["b", "g", "r", "c", "m", "k", "w"]
    line_type = [":", "-.", "--", "/"]
    plot_color_markers = [
        f"{c}{m}" for c, m in itertools.product(plot_colors, plot_markers)
    ]

    def __init__(self) -> None:
        random.Random(13).shuffle(self.plot_color_markers)

    def color_marker_lines(self, line_type="--"):
        return [f"{m}{line_type}" for m in self.plot_color_markers]


PlotUtil = _PlotUtil()


def check_ax(ax=None, **kwargs):
    """
    check if axis exist if not create new figure with one axis
    """
    args = kwargs.copy()
    args.setdefault("figsize", (16, 9))
    if ax is None:
        f, ax = plt.subplots(1, 1, **args)
    else:
        f = ax.get_figure()

    return f, ax


def update_dict(_dic: dict, **defaults):
    """
    set default values for values given in defaults
    """
    if _dic is None:
        return defaults
    else:
        for k, v in defaults.items():
            _dic.setdefault(k, v)
        return _dic


class PlotHelper:
    @classmethod
    def create(cls, ax: plt.Axes = None, **fig_kw):
        return cls(ax, **fig_kw)

    def __init__(self, ax: plt.Axes = None, **fig_kw):
        if ax is None:
            fig, ax = plt.subplots(nrows=1, ncols=1, **fig_kw)
            self._fig = fig
            self._ax = ax
        else:
            self._ax = ax
            self._fig = ax.get_figure()

        self._plot_data: pd.DataFrame = pd.DataFrame()

    def savefig(self, fname, transparent=None, **kwargs):
        self.fig.savefig(fname, transparent=transparent, **kwargs)
        return self

    @property
    def fig(self) -> plt.Figure:
        return self._fig

    @property
    def ax(self) -> plt.Axes:
        return self._ax

    @property
    def plot_data(self) -> pd.DataFrame:
        return self._plot_data


class PlotAttrs:
    """
    PlotAttrs is a singleton guaranteeing unique plot parameters
    """

    class __PlotAttrs:
        plot_marker = [".", "*", "o", "v", "1", "2", "3", "4"]
        plot_color = ["b", "g", "r", "c", "m", "y", "k", "w"]

        def __init__(self):
            pass

    instance: object = None

    def __init__(self):
        if not PlotAttrs.instance:
            PlotAttrs.instance = PlotAttrs.__PlotAttrs()
        self.idx_m = -1
        self.idx_c = -1

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get_marker(self) -> str:
        ret = self.instance.plot_marker[self.idx_m]
        self.idx_m += 1
        if self.idx_m >= len(self.instance.plot_marker):
            self.idx_m = 0
        return ret

    def get_color(self) -> str:
        ret = self.instance.plot_color[self.idx_c]
        self.idx_c += 1
        if self.idx_c >= len(self.instance.plot_color):
            self.idx_c = 0
        return ret

    def reset(self):
        self.idx_c = 0
        self.idx_m = 0


if __name__ == "__main__":
    print(list(PlotUtil.color_marker_lines()))
