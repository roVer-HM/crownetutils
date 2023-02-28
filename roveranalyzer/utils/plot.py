from __future__ import annotations

import itertools
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import Any, ContextManager, List, Protocol, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba_array
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

import roveranalyzer.utils.logging as _log
import roveranalyzer.utils.styles as Styles
from roveranalyzer.simulators.vadere.plots.scenario import VaderScenarioPlotHelper

logger = _log.logger


class FigureSaver:
    """Interface to save figures in some way"""

    @staticmethod
    def FIG(obj: FigureSaver | None, default=None) -> FigureSaver:
        """Provide default implementation if not set """
        if obj is None:
            if default is None:
                return FigureSaverSimple()
            else:
                return default
        else:
            return obj

    def __init__(self) -> None:
        ...

    def __call__(self, figure, *args: Any, **kwargs):
        raise NotImplementedError()

    def __enter__(self, *arg, **kwargs):
        raise NotImplementedError()

    def __exit__(self, exception_type, exception_value, exception_traceback):
        raise NotImplementedError()


class FigureSaverSimple(FigureSaver):
    def __init__(
        self, override_base_path: str | None = None, figure_type: str | None = None
    ):
        self.override_base_path = override_base_path
        self.next_name = None
        self.figure_type = figure_type

    def with_name(self, name):
        self.next_name = name
        return self

    def __call__(self, figure, *args: Any, **kwargs):
        if len(args) < 1 and self.next_name is None:
            raise TypeError("Expected argument for path")
        if self.next_name is None:
            path = args[0]
        else:
            path = self.next_name
            self.next_name = None
        if self.override_base_path is not None:
            if os.path.isabs(path):
                logger.warn(
                    "FigureSaver provides base path but absolute figure path provided. Use override path"
                )
            path = os.path.join(self.override_base_path, os.path.basename(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.figure_type is not None:
            base, ext = os.path.splitext(path)
            if self.figure_type != ext:
                logger.info(f"override figure type from {ext} to {self.figure_type}")
            path = f"{base}{self.figure_type}"
        figure.tight_layout()
        figure.savefig(path)

    def __enter__(self, *arg, **kwargs):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        # nothing todo. No context to close because caller provides figure for saving
        pass


class FigureSaverPdfPages:
    @classmethod
    @contextmanager
    def withSaver(cls, path, **kwargs) -> ContextManager[FigureSaverPdfPages]:
        try:
            saver: FigureSaverPdfPages = cls(path, **kwargs)
            yield saver
        finally:
            saver.pdf.close()

    def __init__(self, pdf: PdfPages | str, **kwargs):
        if isinstance(pdf, PdfPages):
            self.pdf = pdf
        else:
            self.pdf = PdfPages(pdf, **kwargs)
        self.__entered = False

    def __call__(self, figure, *args: Any, **kwargs):
        self.pdf.savefig(figure)

    def __enter__(self, *arg, **kwargs):
        if self.__entered:
            raise RuntimeError("object already opened.")
        self.__entered = True
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.pdf.close()
        self.__entered = False


def matplotlib_set_latex_param() -> matplotlib.RcParams:
    old = Styles.load_matplotlib_style(Styles.STYLE_TEX)
    p = "\n".join(
        [  # plots will use this preamble
            r"\usepackage[utf8]{inputenc}",
            r"\usepackage[T1]{fontenc}",
            r"\usepackage[detect-all,round-mode=places,tight-spacing=true]{siunitx}",
        ]
    )
    matplotlib.rcParams.update({"pgf.preamble": p, "tex.latex.preamble": p})
    return old


def plt_rc_same(rc: None | dict = None, size="xx-large"):
    _rc = {
        "axes.titlesize": size,
        "axes.labelsize": size,
        "xtick.labelsize": size,
        "ytick.labelsize": size,
        "legend.fontsize": size,
        "figure.titlesize": size,
    }
    if rc is None:
        rc = _rc
    else:
        rc.update(**_rc)
    return rc


def paper_rc(tick_labelsize="xx-large", rc=None, **kw):
    _rc = {
        "axes.titlesize": "xx-large",
        "xtick.labelsize": tick_labelsize,
        "ytick.labelsize": tick_labelsize,
        "legend.fontsize": "xx-large",
    }
    if rc is None:
        rc = _rc
        rc.update(**kw)
    else:
        for k, v in _rc.items():
            if k not in rc:
                rc[k] = v
        rc.update(**kw)
    return rc


def rename_legend(ax: plt.Axes, rename: dict | None = None, **kwargs) -> plt.Axes:
    """[deprecated] added to PlotUtil class"""
    PlotUtil.rename_legend(ax, rename, **kwargs)


def tight_ax_grid(nrows, ncols, **kwds):
    kwds.setdefault("sharey", "all")
    kwds.setdefault("sharex", "all")
    fig, axes = plt.subplots(nrows, ncols, **kwds)
    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r][c]
            if r % nrows == 0 and c % ncols == 0:
                # bottom left all axis
                pass
            elif r % nrows == 0:
                # left only y axis
                pass
            elif c % ncols == 0:
                # bootom only x axis
                pass
            else:
                # center axis nothing
                # ax.set_xticklabels([])
                pass
    return fig, axes


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
    _hatch_patterns = ("||", "--", "++", "x", "\\", "*", "|", "-", "+")
    _plot_markers = ["o", "x", "*", ".", "v", "1", "2", "3", "4"]
    _plot_colors = ["b", "g", "r", "c", "m", "k"]
    _line_types = ["-", ":", "-.", "--"]
    _plot_color_markers = [
        f"{c}{m}" for c, m in itertools.product(_plot_colors, _plot_markers)
    ]
    _plot_color_lines = [
        f"{c}{l}" for c, l in itertools.product(_plot_colors, _line_types)
    ]

    def __init__(self) -> None:
        random.Random(13).shuffle(self._plot_color_markers)
        random.Random(13).shuffle(self._plot_color_lines)
        self.ax_provider = self._check_ax

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

    def append_title(self, ax: plt.Axes, *, prefix="", suffix=""):
        """Append string at front or back of the axes title

        Args:
            ax (plt.Axes): Axes object to change title
            prefix (str, optional): Prefix string to append. Defaults to "".
            suffix (str, optional): Suffix string to append. Defaults to "".
        """
        _title = ax.get_title()
        prefix = prefix if len(prefix) < 1 else f"{prefix.strip()} "
        _title = _title if len(suffix) < 1 else f"{_title.strip()} "
        ax.set_title(f"{prefix}{_title}{suffix}")

    def par(self, key, default=None):
        return plt.rcParams.get(key, default)

    def ecdf(
        self, data: pd.Series | pd.DataFrame, ax: plt.Axes | None = None, **kwargs
    ):
        fig, ax = self.check_ax(ax)
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]  # first column
        _x = data.sort_values()
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, drawstyle="steps-pre", **kwargs)
        ax.set_ylabel("density")
        return ax

    def color_marker_lines(self, line_type="--"):
        return [f"{m}{line_type}" for m in self._plot_color_markers]

    def color_lines(self, line_type: Union[str, List[str], str] = None, cycle=True):
        if line_type is None:
            lines = self._plot_color_lines
        elif type(line_type) == list:
            lines = [
                f"{c}{l}" for c, l in itertools.product(self._plot_colors, line_type)
            ]
            random.Random(13).shuffle(lines)
        elif type(line_type) == str:
            lines = [f"{m}{line_type}" for m in self._plot_color_markers]
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

    def df_to_table(
        self, df: pd.DataFrame, ax: plt.Axes | None = None, title: str | None = None
    ):
        fig, ax = self.check_ax(ax)
        fig.patch.set_visible(False)
        ax.axis("off")
        t = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
        t.auto_set_font_size(False)
        t.set_fontsize(11)
        t.auto_set_column_width(col=(list(range(df.shape[1]))))
        [c.set_height(0.06) for c in t.get_celld().values()]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        if title is not None:
            ax.set_title(title)
        return fig, ax

    def fig_to_pdf(self, path, figures: List[plt.figure]):
        with PdfPages(path) as pdf:
            for f in figures:
                pdf.savefig(f)

    def check_ax(self, ax=None, **kwargs):
        """
        check if axis exist if not create new figure with one axis
        """
        return self.ax_provider(ax, **kwargs)

    def _check_ax(self, ax=None, **kwargs):
        args = kwargs.copy()
        args.setdefault("figsize", (16, 9))
        if ax is None:
            f, ax = plt.subplots(1, 1, **args)
        else:
            f = ax.get_figure()

        return f, ax

    @property
    def color_marker_lines_cycle(self):
        return itertools.cycle(self.color_marker_lines())

    def rename_legend(
        self, ax: plt.Axes, rename: dict | None = None, **kwargs
    ) -> plt.Axes:
        rename = {} if rename is None else rename
        rename.update(kwargs)
        for t in ax.get_legend().texts:
            if t.get_text() in rename:
                t.set_text(rename[t.get_text()])
        return ax

    def get_vadere_legal_cells(
        self,
        scenario: VaderScenarioPlotHelper,
        xy_slices: Tuple[slice, slice],
        c: float | Tuple[float, float] = 5.0,
    ):
        """Creates free and obstacle covered cell indexes within the provided
            rectangle area

        Args:
            scenario (VaderScenarioPlotHelper): Wrapper object of Vadere scenario file
            xy_slices (Tuple[slice, slice]): rectangle area slice to use
            c (float|Tuple[float, float], optional): Cell size if not provided as as slice step. Defaults to 5.0.
        """

        _covered = []
        _free = []
        _x, _y = xy_slices
        c = (c, c) if isinstance(c, float) else c
        _x_step = c[0] if _x.step is None else _x.step
        _y_step = c[1] if _y.step is None else _y.step
        obs: List[Polygon] = [
            scenario.scenario.shape_to_list(s["shape"], to_shapely=True)
            for s in scenario.scenario.obstacles
        ]
        for x in np.arange(_x.start, _x.stop, _x_step):
            for y in np.arange(_y.start, _y.stop, _y_step):
                idx = (x, y)
                cell = Polygon(
                    [
                        [x, y],
                        [x + _x_step, y],
                        [x + _x_step, y + _y_step],
                        [x, y + _y_step],
                        [x, y],
                    ]
                )
                for _o in obs:
                    if _o.covers(cell):
                        _covered.append(idx)
                        break
                if (len(_covered) == 0) or (_covered[-1] != idx):
                    _free.append(idx)

        _covered = pd.MultiIndex.from_tuples(_covered, names=["x", "y"])
        _free = pd.MultiIndex.from_tuples(_free, names=["x", "y"])
        return _free, _covered

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

    def mult_locator(self, axis, major, minor=None):
        axis.set_major_locator(MultipleLocator(major))
        axis.set_minor_locator(MultipleLocator(minor))
        _which = "major" if minor is None else "both"
        _axis = axis.axis_name
        axis.axes.grid(True, _which, _axis)

    def add_colorbar(self, im, aspect=20, pad_fraction=0.5, **kwargs):
        """Add a vertical color bar to an image plot."""
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    def fill_between(
        self,
        data: pd.DataFrame,
        x=None,
        val=None,
        fill_val=None,
        fill_alpha=0.2,
        *,
        plot_lbl: str | None = None,
        line_args: dict | None = None,
        fill_args: dict | None = None,
        ax: plt.Axes | None = None,
        **kwds,
    ) -> plt.Axes:
        """Create error bar plot with filled area of same color with reduced alpha"""
        if x is None:
            # assume first level as x-axes
            x = data.index.get_level_values(0)
        elif isinstance(x, str):
            x = data[x]

        if val is None:
            # assume first column if not set
            val = data.iloc[:, 0]
        elif isinstance(val, str):
            val = data[val]

        if fill_val is None:
            # assume symmetric additive bounds based on second column
            fill_val = data.iloc[:, 1]
            l_bound = val - fill_val
            u_bound = val + fill_val
        elif isinstance(fill_val, str):
            # assume symmetric additive bounds based on column
            fill_val = data[fill_val]
            l_bound = val - fill_val
            u_bound = val + fill_val
        elif isinstance(fill_val, list):
            # use two sided bounds with absolute values
            l_bound = data.loc[:, fill_val[0]]
            u_bound = data.loc[:, fill_val[1]]

        fig, ax = self.check_ax(ax)
        line = ax.plot(x, val, **({} if line_args is None else line_args))[-1]
        if plot_lbl is not None:
            line.set_label(plot_lbl)
        ret_fill = ax.fill_between(
            x,
            l_bound,
            u_bound,
            alpha=fill_alpha,
            color=line.get_color(),
            interpolate=True,
            **({} if fill_args is None else fill_args),
        )
        return ax, line, ret_fill


PlotUtil = _PlotUtil()


def with_axis(method):
    @wraps(method)
    def with_axis_impl(self, *method_args, **method_kwargs):
        if "ax" not in method_kwargs:
            _, ax = PlotUtil.check_ax(None)
            method_kwargs.setdefault("ax", ax)
        elif "ax" in method_kwargs and method_kwargs["ax"] is None:
            _, ax = PlotUtil.check_ax()
            method_kwargs["ax"] = ax
        return method(self, *method_args, **method_kwargs)

    return with_axis_impl


def savefigure(method):
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
            elif isinstance(savefig, str):
                os.makedirs(os.path.dirname(os.path.abspath(savefig)), exist_ok=True)
                logger.info(f"save figure: {savefig}")
                fig.savefig(savefig)
            else:
                # assume some callable
                savefig(fig)
        return fig, ax

    return savefigure_impl


def plot_decorator(method):
    @wraps(method)
    def _plot_decorator(self, *method_args, **method_kwargs):
        if self.plot_wrapper is not None:
            return self.plot_wrapper(method, self, *method_args, **method_kwargs)
        else:
            return method(self, *method_args, **method_kwargs)

    return _plot_decorator


def tex_1col_fig(ratio=16 / 9, *arg, **kwargs):
    return plt.subplots(*arg, **kwargs, figsize=(5, 5 / ratio))


def tex_2col_fig(ratio=16 / 9, *arg, **kwargs):
    return plt.subplots(*arg, **kwargs, figsize=(18, 18 / ratio))


@contextmanager
def empty_fig(title) -> ContextManager[plt.figure]:
    fig, ax = PlotUtil.check_ax()
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
