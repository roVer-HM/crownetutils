from __future__ import annotations

import itertools
import os
import random
from contextlib import contextmanager
from functools import wraps
from tkinter import N
from typing import Any, ContextManager, List, Union

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba_array
from shapely.geometry import Polygon

import roveranalyzer.utils.logging as _log

logger = _log.logger


def matplotlib_set_latex_param():
    sns.set(font_scale=1.0, rc={"text.usetex": True})
    sns.set_style("whitegrid")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pgf.texsystem"] = "pdflatex"
    matplotlib.rcParams.update(
        {
            "font.family": "serif",
            # "font.size": 18,
            "axes.labelsize": 20,
            "axes.titlesize": 22,
            "legend.fontsize": 18,
            "figure.titlesize": 24,
            "pgf.preamble": "\n".join(
                [  # plots will use this preamble
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage[detect-all,round-mode=places,tight-spacing=true]{siunitx}",
                ]
            ),
        }
    )

    p = "\n".join(
        [  # plots will use this preamble
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,round-mode=places,tight-spacing=true]{siunitx}",
        ]
    )
    matplotlib.rc("text.latex", preamble=p)
    matplotlib.rcParams["text.usetex"] = True


def paper_rc(tick_labelsize="xx-large", **kw):
    rc = {
        "axes.titlesize": "xx-large",
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize,
        "legend.fontsize": "xx-large",
    }
    rc.update(**kw)
    return rc


def remove_seaborn_legend_title(ax: plt.Axes):
    ax.get_legend().set_title(None)
    return ax


def rename_legend(ax: plt.Axes, rename: dict | None = None, **kwargs) -> plt.Axes:
    rename = {} if rename is None else rename
    rename.update(kwargs)
    for t in ax.get_legend().texts:
        if t.get_text() in rename:
            t.set_text(rename[t.get_text()])
    return ax


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


class StyleMap:
    """Provide n different colored styles for some line plot
    where each style is mapped to some key. If more than `n`
    different keys are quired an exception is raised.
    """

    def __init__(self, cmap="tab20", n=20, **default_style):
        self._style_map = {}
        self._default = dict(default_style)
        self._color_set: list = plt.get_cmap(cmap)(np.linspace(0.0, 1.0, n))

    def _new_style(self):
        if len(self._style_map) == len(self._color_set):
            raise ValueError("Number of Styles reached")
        s = {}
        s.update(self._default)
        s["color"] = self._color_set[len(self._style_map)]
        return s

    def get_style(self, key: Any) -> dict:
        if key not in self._style_map:
            s = self._new_style()
            self._style_map[key] = s
        return self._style_map[key]


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

    def contour_two_slope_colors(
        self, norm: TwoSlopeNorm, n_colors, c_map="coolwarm", mid_white: bool = False
    ):
        lvl = plt.MaxNLocator(nbins=n_colors)
        levels = lvl.tick_values(norm.vmin, norm.vmax)
        colors = plt.get_cmap(c_map)(norm(levels))

        if mid_white:
            colors[(levels == 0.0).argmax()] = [1.0, 1.0, 1.0, 1.0]
            # colors[(levels==.0).argmax()-1] = [1., 1., 1., 1.]
        return levels, colors

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

    def cell_to_tex(
        self, polygons: List[Polygon] | pd.MultiIndex, c=5.0, fd=None, attr=None, **kwds
    ):
        ret = ""
        if attr is None:
            attr = ""
        else:
            attr = ", ".join(attr)
        attr += ", ".join([f"{k}={v}" for k, v in kwds.items()])

        if isinstance(polygons, pd.MultiIndex):
            polygons = [
                Polygon([[x, y], [x + c, y], [x + c, y + c], [x, y + c], [x, y]])
                for x, y in polygons
            ]

        for cell_id, p in enumerate(polygons):
            coords = list(p.exterior.coords)
            ret += f"% {cell_id} Cell {coords[0][0]}{coords[0][1]}\n"
            ret += f"\draw[{attr}] "
            for i, (_x, _y) in enumerate(coords):
                if i == (len(coords) - 1):
                    ret += f"({_x}, {_y});"
                else:
                    ret += f"({_x}, {_y}) to "
            ret += "\n"
        if fd is None:
            return ret
        else:
            with open(fd, "w", encoding="utf-8") as f:
                f.write(ret)


PlotUtil = _PlotUtil()


def tex_1col_fig(ratio=16 / 9, *arg, **kwargs):
    return plt.subplots(*arg, **kwargs, figsize=(5, 5 / ratio))


def tex_2col_fig(ratio=16 / 9, *arg, **kwargs):
    return plt.subplots(*arg, **kwargs, figsize=(18, 18 / ratio))


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


@contextmanager
def empty_fig(title) -> ContextManager[plt.figure]:
    fig, ax = check_ax()
    ax.axis("off")
    fig.text(0.5, 0.5, title, ha="center", va="center")
    yield fig
    plt.close(fig)


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
