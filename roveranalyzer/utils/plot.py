import itertools
import os
import random
from functools import wraps
from typing import List, Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

import roveranalyzer.utils.logging as _log

logger = _log.logger


def matplotlib_set_latex_param():
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 24,
            "legend.fontsize": 24,
            "figure.titlesize": 28,
        }
    )
    matplotlib.rcParams["text.usetex"] = True


class Style:
    def __init__(self) -> None:
        self.font_dict = {
            "title": {"fontsize": 24},
            "xlabel": {"fontsize": 20},
            "ylabel": {"fontsize": 20},
            "legend": {"size": 20},
            "tick_size": 16,
        }
        self.create_legend = True


class _PlotUtil:
    hatch_patterns = ("||", "--", "++", "x", "\\", "*", "|", "-", "+")
    plot_markers = ["o", "x", "*", ".", "v", "1", "2", "3", "4"]
    plot_colors = ["b", "g", "r", "c", "m", "k"]
    line_types = ["-", ":", "-.", "--"]
    plot_color_markers = [
        f"{c}{m}" for c, m in itertools.product(plot_colors, plot_markers)
    ]
    plot_color_lines = [
        f"{c}{l}" for c, l in itertools.product(plot_colors, line_types)
    ]

    def __init__(self) -> None:
        random.Random(13).shuffle(self.plot_color_markers)
        random.Random(13).shuffle(self.plot_color_lines)

    def color_marker_lines(self, line_type="--"):
        return [f"{m}{line_type}" for m in self.plot_color_markers]

    def color_lines(self, line_type: Union[str, List[str], str] = None, cycle=True):
        if line_type is None:
            lines = self.plot_color_lines
        elif type(line_type) == list:
            lines = [
                f"{c}{l}" for c, l in itertools.product(self.plot_colors, line_type)
            ]
            random.Random(13).shuffle(lines)
        elif type(line_type) == str:
            lines = [f"{m}{line_type}" for m in self.plot_color_markers]
        else:
            raise ValueError("expected None, list of strings or string")
        if cycle:
            return itertools.cycle(lines)
        else:
            return lines

    def equalize_axis(self, axes: List[plt.Axes], axis="y"):
        if axis == "y":
            max_ = max([i.get_yticks().max() for i in axes])
            min_ = min([i.get_yticks().min() for i in axes])
            for ax in axes:
                ax.set_ylim([min_, max_])
        else:
            max_ = max([i.get_xticks().max() for i in axes])
            min_ = min([i.get_xticks().min() for i in axes])
            for ax in axes:
                ax.set_xlim([min_, max_])

    def df_to_table(self, df: pd.DataFrame, ax: plt.Axes):
        t = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(11)
        t.auto_set_column_width(col=(list(range(df.shape[1]))))
        [c.set_height(0.04) for c in t.get_celld().values()]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    def fig_to_pdf(self, path, figures: List[plt.figure]):
        with PdfPages(path) as pdf:
            for f in figures:
                pdf.savefig(f)

    @property
    def color_marker_lines_cycle(self):
        return itertools.cycle(self.color_marker_lines())

    def with_axis(self, method):
        @wraps(method)
        def with_axis_impl(self, *method_args, **method_kwargs):
            if "ax" not in method_kwargs:
                _, ax = check_ax(None)
                method_kwargs.setdefault("ax", ax)
            return method(self, *method_args, **method_kwargs)

        return with_axis_impl

    def savefigure(self, method):
        @wraps(method)
        def savefigure_impl(self, *method_args, **method_kwargs):
            savefig = None
            if "savefig" in method_kwargs:
                savefig = method_kwargs["savefig"]
                del method_kwargs["savefig"]
            fig, ax = method(self, *method_args, **method_kwargs)
            if savefig is not None:
                if isinstance(savefig, PdfPages):
                    savefig.savefig(fig)
                else:
                    os.makedirs(
                        os.path.dirname(os.path.abspath(savefig)), exist_ok=True
                    )
                    logger.info(f"save figure: {savefig}")
                    fig.savefig(savefig)
            return fig, ax

        return savefigure_impl

    def plot_decorator(self, method):
        @wraps(method)
        def _plot_decorator(self, *method_args, **method_kwargs):
            if self.plot_wrapper is not None:
                return self.plot_wrapper(method, self, *method_args, **method_kwargs)
            else:
                return method(self, *method_args, **method_kwargs)

        return _plot_decorator


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
