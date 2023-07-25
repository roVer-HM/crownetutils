""" Miscellaneous utility functions and classes to simplify common plotting tasks.

This contains default configurations of the matplotrc, dataframe to table, and 
decorator functions to inject axes and save paths. 

`PlotUtl_` is the main class that combines plot related helpers. Specialized plot
classes for specific simulation runs can inherit from this class to reuse plot 
functionalities. 

"""
from __future__ import annotations

import itertools
import os
import random
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    List,
    Protocol,
    Tuple,
    Union,
)

import matplotlib
import matplotlib.patches as pltPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colorbar import Colorbar
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgba_array
from matplotlib.image import AxesImage
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

import crownetutils.utils.logging as _log
import crownetutils.utils.styles as Styles
from crownetutils.vadere.plot.topgraphy_plotter import VadereTopographyPlotter

logger = _log.logger


def percentile(n: int) -> Callable[[Any], Any]:
    """Function to generate a numpy based percentile function

    Args:
        n (int): Percentile to compute, which must be between 0 and 100 inclusive.
    Returns:
        Callable[[Any], Any]: Function that compute the n-th percentile of the provided data.
    """

    def percentile_(x):
        if not x.empty:
            return np.percentile(x, n)
        else:
            return np.nan

    percentile_.__name__ = f"p{n}"
    return percentile_


def with_axis(func):
    """Decorator that injects an keyword argument 'ax' of the
    type `plt.Axes` if missing.

    Args:
        func (Callable): Function to be decorated

    Returns:
        Callable: Decorated (i.e. extended) function
    """

    @wraps(func)
    def with_axis_impl(self, *func_args, **func_kwargs):
        if "ax" not in func_kwargs:
            _, ax = PlotUtil.check_ax(None)
            func_kwargs.setdefault("ax", ax)
        elif "ax" in func_kwargs and func_kwargs["ax"] is None:
            _, ax = PlotUtil.check_ax()
            func_kwargs["ax"] = ax
        return func(self, *func_args, **func_kwargs)

    return with_axis_impl


def savefigure(func):
    """Decorator that looks for a keyword argument 'savefig'.
    todo::
    Args:
        func (Callable): Function to be decorated

    Returns:
        Callable: Decorated (i.e. extended) function
    """

    @wraps(func)
    def savefigure_impl(self, *func_args, **func_kwargs):
        savefig = None
        if "savefig" in func_kwargs:
            savefig = func_kwargs["savefig"]
            del func_kwargs["savefig"]
        fig, ax = func(self, *func_args, **func_kwargs)
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


class FigureSaver:
    """Interface to save figures in some way"""

    @staticmethod
    def FIG(obj: FigureSaver | None, default=None) -> FigureSaver:
        """Provide default implementation if not set"""
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


class NullSaver(FigureSaver):
    def __call__(self, figure, *args: Any, **kwargs):
        return

    def __enter__(self, *arg, **kwargs):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        return


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


class PlotUtil_:
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

    def get_default_color_cycle(self):
        """Default color cycle defined in currently set rcParams"""
        return plt.rcParams["axes.prop_cycle"].by_key()["color"]

    def par(self, key: str, default: Any = None) -> Any:
        """Matplolib rcParams wrapper to access rcParams or return default. This is a read only access!

        Args:
            key (str): rcParmas key to access
            default (Any, optional): Default value. Defaults to None.

        Returns:
            Any: rcParameter for key or provided default. Can be None.
        """
        return plt.rcParams.get(key, default)

    @with_axis
    def ecdf(
        self, data: pd.Series | pd.DataFrame, ax: plt.Axes | None = None, **kwargs
    ) -> plt.Axes:
        """Create empirical copulative density function (ECDF) of provided data.

        Args:
            data (pd.Series | pd.DataFrame): Data used. If Dataframe use first column
            ax (plt.Axes | None, optional): Provided axes for plotting. New object inject via `@with_axis` if None. Defaults to None.

        Returns:
            plt.Axes:
        """
        if isinstance(data, pd.DataFrame):
            data = data.iloc[:, 0]  # first column
        _x = data.sort_values()
        _y = np.arange(len(_x)) / float(len(_x))
        ax.plot(_x, _y, drawstyle="steps-pre", **kwargs)
        ax.set_ylabel("density")
        return ax

    def color_marker_lines(self, line_type="--") -> List[str]:
        """Create color/marker/line_type string for provided line type.

        Example:
            For line_type='--'
            [bo--, go--, ro--, b*--, g*--, r*--]
            'black line with spaces ----o----o---and non-filled circle marker

        Args:
            line_type (str, optional): Legal line type of matplotlib. E.g. [-, :, -., --].Defaults to "--".

        Returns:
            List[str]: List of color/marker/line strings for plot line formatting.
        """
        return [f"{m}{line_type}" for m in self._plot_color_markers]

    def color_lines(
        self, line_type: str | List[str] | None = None, cycle: bool = True
    ) -> List[str] | Iterable[str]:
        """Create color/marker string for line formatting in matpltolib.

        Args:
            line_type (str | List[str] | None, optional): Line. Defaults to None.
            cycle (bool, optional): _description_. Defaults to True.

        Raises:
            ValueError: _description_

        Returns:
            List[str]|Iterable[str]: _description_
        """
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
        """Equalize x or y ax limit to min/max of all provided axis.

        This ensures that multiple axes show the same area and are comparable.

        Args:
            axes (List[plt.Axes]): List of axes to equalize
            axis (str, optional): axis identifier. Defaults to "y".
        """
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
    ) -> Tuple[plt.Figuer, plt.Axes, plt.Table]:
        """Save columns of dataframe as matplotlib table.

        Args:
            df (pd.DataFrame):
            ax (plt.Axes | None, optional): Axes to which to add table. If None new one is created. Defaults to None.
            title (str | None, optional): title of table. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]:
        """
        fig, ax = self.check_ax(ax)
        fig.patch.set_visible(False)
        ax.axis("off")
        t = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            loc="center",
            bbox=[0.0, 0.0, 1.0, 1.0],
        )
        t.auto_set_font_size(False)
        t.set_fontsize(11)
        t.auto_set_column_width(col=(list(range(df.shape[1]))))
        [c.set_height(0.2) for c in t.get_celld().values()]
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        if title is not None:
            ax.set_title(title)
        return fig, ax, t

    def fig_to_pdf(self, path, figures: List[plt.figure], close_figures: bool = False):
        """Save list of figures into one pdf file. Close figure object at the end if set.

        Args:
            path (str): Path of pdf.
            figures (List[plt.figure]): Figures to save to pdf.
            close_figures (bool): If yes close figures after save. Default False.
        """
        with PdfPages(path) as pdf:
            for f in figures:
                pdf.savefig(f)
                if close_figures:
                    plt.close(f)

    def check_ax(
        self, ax: plt.Axes | None = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Return figure, ax pair of given `ax` based on configured `ax_provider`. If ax is None create a new plot with one axes.

        Args:
            ax (plt.Axes | None, optional): _description_. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]:
        """
        return self.ax_provider(ax, **kwargs)

    def _check_ax(
        self, ax: plt.Axes | None = None, **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Return figure, ax pair of given `ax`. If ax is None create a new plot with one axes. This is the default `ax_provider`
        if nothing else was set.

        Args:
            ax (plt.Axes|None, optional): If None create new object otherwise just return proved axes without modification. Defaults to None.

        Returns:
            Tuple[plt.Figure, plt.Axes]
        """
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
        scenario: VadereTopographyPlotter,
        xy_slices: Tuple[slice, slice],
        c: float | Tuple[float, float] = 5.0,
    ) -> Tuple[pd.MultiIndex, pd.MultiIndex]:
        """Creates free and obstacle covered cell indexes within the provided rectangle area

        Args:
            scenario (VaderScenarioPlotHelper): Wrapper object of Vadere scenario file
            xy_slices (Tuple[slice, slice]): rectangle area slice to use
            c (float|Tuple[float, float], optional): Cell size if not provided as as slice step. Defaults to 5.0.

        Returns:
            Tuple[pd.MultiIndex, pd.MultiIndex]: free and covered cells.
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
        self,
        polygons: List[Polygon] | pd.MultiIndex,
        c=5.0,
        fd=None,
        attr: None | Dict[str, Any] = None,
        **kwds,
    ) -> str | None:
        """Create tikz string of provided polygons and save or return result. In polygons is a pd.MultiIndex
        create polygons, i.e. square cells with a length of `c`, first.

        Args:
            polygons (List[Polygon] | pd.MultiIndex): List of Polygons or input to create square cells
            c (float, optional): Cell side length in case `polygons` is an index. Defaults to 5.0.
            fd (str, optional): File path to save result to or None if result should be returned. Defaults to None.
            attr (None | Dict[str,Any], optional): Tikz attributes for `\draw` command. Defaults to None.

        Returns:
            str|None: String of `\draw` command or nothing if saved to file.
        """

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

    def mult_locator(self, axis: plt.Axes, major: float, minor: float | None = None):
        """Append major or major and minor axis locators to provided axes.

        Args:
            axis (plt.Axes): Axes.
            major (float): Tick at each integer multiple
            minor (float | None, optional): Tick at each integer multiple. If None do not display. Defaults to None.
        """
        axis.set_major_locator(MultipleLocator(major))
        axis.set_minor_locator(MultipleLocator(minor))
        _which = "major" if minor is None else "both"
        _axis = axis.axis_name
        axis.axes.grid(True, _which, _axis)

    def add_colorbar(
        self, im: AxesImage, aspect: float = 20, pad_fraction: float = 0.5, **kwargs
    ) -> Colorbar:
        """Add a vertical `Colorbar` on the right side to an image plot

        Args:
            im (AxesImage): Image to which the color bar is added.
            aspect (float, optional): Aspect ratio for color bar. Defaults to 20.
            pad_fraction (float, optional): padding fraction. Defaults to 0.5.

        Returns:
            Colorbar: Create colorbar
        """
        divider = axes_grid1.make_axes_locatable(im.axes)
        width = axes_grid1.axes_size.AxesY(im.axes, aspect=1.0 / aspect)
        pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
        current_ax = plt.gca()
        cax = divider.append_axes("right", size=width, pad=pad)
        plt.sca(current_ax)
        return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    def append_bin(
        self,
        data: pd.DataFrame,
        idx_name: str = "time",
        bin_size: float = 1.0,
        start: float | None = None,
        end: float | None = None,
        *,
        closed: str = "right",
        columns: None | List[str] = None,
        agg: None | List[Any] = None,
    ) -> pd.DataFrame:
        """Append interval index based on the index 'idx_name'. If aggregation functions are provided
        use interval index as group key and evaluate the aggregation function.

        Args:
            data (pd.DataFrame): Input data.
            idx_name (str, optional): Name of column or index to used for interval range. Defaults to "time".
            bin_size (float, optional): Size of interval. Defaults to 1.0.
            start (float | None, optional): Start value for interval range. If None use data minimum. Defaults to None.
            end (float | None, optional): End value of interval range. If None use data maximum. Defaults to None.
            columns (None | List[str], optional): Column filter applied before aggregation is applied. Only used if agg is not None. Defaults to None.
            agg (None | List[Any], optional): List of aggregation function names or callables used in groupby.agg([....]). Defaults to None.

        Raises:
            ValueError: idx_name must be present in columns or as an index value.

        Returns:
            pd.DataFrame: Input with appended interval index or aggregation result over interval index.
        """
        if idx_name in data.columns:
            data = data.reset_index(drop=True).set_index(idx_name).sort_index()
            idx = data.index.get_level_values(idx_name)
        elif idx_name in data.index.names:
            idx = data.index.get_level_values(idx_name)
        else:
            raise ValueError("No index found with name {idx_name}")
        start = idx.min() if start is None else start
        end = idx.max() + bin_size if end is None else end
        bins = pd.interval_range(start=start, end=end, freq=bin_size, closed=closed)
        data["bin"] = pd.cut(idx, bins, right=closed == "right")
        if agg is not None:
            _n = data.index.names
            data = data.reset_index().set_index([*_n, "bin"]).sort_index()
            if columns is not None:
                data = data[columns]
            data = data.groupby("bin").agg(agg).dropna()
            data = data.set_axis([f"{a}_{b}" for a, b in data.columns], axis=1)
            data["bin_left"] = data.index.to_series().apply(lambda x: x.left)
            data["bin_right"] = data.index.to_series().apply(lambda x: x.right)
        return data

    @with_axis
    def fill_between(
        self,
        data: pd.DataFrame,
        x: str | None = None,
        val: str | None = None,
        fill_val=None,
        fill_alpha=0.2,
        *,
        plot_lbl: str | None = None,
        line_args: dict | None = None,
        fill_args: dict | None = None,
        ax: plt.Axes | None = None,
        **kwds,
    ) -> plt.Axes:
        """Generate plot with the actual data and a shaded 'fill between area' with the same color.

        If parameters `x`, `val`, `fill_val` are None and data as exactly 4 columns the columns are
        read as [x, val, l_bound, u_bound]. Otherwise see argument description.

        Args:
            data (pd.DataFrame): Dataframe containing at least the data and optional upper and lower bounds.
            x (str|None, optional): Column name for x axis. If None use first index (index 0) of dataframe. Defaults to None.
            val (str|None, optional): Column name for data. If None use column 0 of dataframe. Defaults to None.
            fill_val (str|List[str]|None, optional):Column name of symmetric upper and lower bound if string.
                                                    If None use column 1 of dataframe. If list use as [lower, upper] bound. Defaults to None.
            fill_alpha (float, optional): Alpha value of area between lower and upper bound. Defaults to 0.2.
            plot_lbl (str | None, optional): Label for data. Defaults to None.
            line_args (dict | None, optional): Kwargs passed to line plot of data values. Defaults to None.
            fill_args (dict | None, optional): Kwargs passed to fill_between plot. Defaults to None.
            ax (plt.Axes | None, optional): Axes object to add line to. If None `with_axis` decorator will inject new object. Defaults to None.

        Returns:
            plt.Axes:
        """
        # Create error bar plot with filled area of same color with reduced alpha
        if all([i is None for i in [x, val, fill_val]]) and data.shape[1] == 4:
            # noting set and exactly 4 columns
            print("found 4 columns. Use as 'x, value, l_bound, u_bound' ")
            x = data.iloc[:, 0]
            val = data.iloc[:, 1]
            l_bound = data.iloc[:, 2]
            u_bound = data.iloc[:, 3]
        else:
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

    def color_box_plot(
        self,
        bp: dict,
        fill_color,
        ax: plt.Axes,
        edge_color="black",
        flier="+",
        flier_size=3,
    ) -> None:
        """Create manual filled box plots based on the `bp` dictionary returned by `pandas.GroubBy.boxplot` function

        Args:
            bp (dict): Dictionary of Box artists
            fill_color (_type_): Color to use to fill the boxes.
            ax (plt.Axes): Axes to add the patches to.
            edge_color (str, optional): Edge color for boxes. Defaults to "black".
            flier (str, optional): Marker for fliers. Defaults to "+".
            flier_size (int, optional): Marker size for fliers. Defaults to 3.
        """
        if isinstance(fill_color, list):
            if len(fill_color) != len(bp["boxes"]):
                raise ValueError("Number of colors des not match number of boxes.")
        else:
            fill_color = [fill_color]

        for idx, color in enumerate(fill_color):
            plt.setp(bp["boxes"][idx], color=edge_color, zorder=2.1)
            plt.setp(bp["medians"][idx], color=edge_color)
            plt.setp(
                bp["fliers"][idx],
                mec=color,
                mfc=color,
                marker=flier,
                markersize=flier_size,
            )
            if "means" in bp and len(bp["means"]) > 0:
                plt.setp(
                    bp["means"][idx],
                    mec=edge_color,
                    mfc=color,
                    marker="*",
                    markersize=flier_size + 2,
                )

            b = bp["boxes"][idx]
            _coords = list(zip(b.get_xdata(), b.get_ydata()))
            _patch = pltPatch.Polygon(_coords, facecolor=color, zorder=2.0)
            ax.add_patch(_patch)

    def merge_legend_patches(self, h, l):
        """Merge provided handles pair-wise using the first label. This is useful to combine
           the labels of fill between plots where the line and the shaded area are combined.

           This method assumes same length of both `h` and `l` as well as even number of items in each.
        Args:
            h (_type_): List of label handles.
            l (_type_): List of label strings.

        Returns:
            _type_: Tuple of handle and label lists.
        """
        l_new = []
        h_new = []
        i = 0
        while i < len(l):
            h_new.append((h[i], h[i + 1]))
            l_new.append(l[i])
            i += 2
        return h_new, l_new


PlotUtil = PlotUtil_()


def tex_1col_fig(
    ratio: float = 16 / 9, *arg, **kwargs
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """Return 1 column wide figure for latex 2 column based template

    Args:
        ratio (float, optional): Figure ratio. Defaults to 16/9.

    Returns:
        tuple[plt.Figure, list[list[plt.Axes]]]: _description_
    """
    return plt.subplots(*arg, **kwargs, figsize=(5, 5 / ratio))


def tex_2col_fig(
    ratio: float = 16 / 9, *arg, **kwargs
) -> tuple[plt.Figure, list[list[plt.Axes]]]:
    """Return 2 colum wide figure for latex 2 column based templates.

    Args:
        ratio (float, optional): Figure ratio. Defaults to 16/9.

    Returns:
        tuple[plt.Figure, list[list[plt.Axes]]]
    """
    return plt.subplots(*arg, **kwargs, figsize=(18, 18 / ratio))


@contextmanager
def empty_fig(title) -> ContextManager[plt.figure]:
    """Create empty figure with centered text field.

    Function ensures that created figure is closed to mitigate warning about to many figures.

    Args:
        title (str): Text to add to figure

    Returns:
        ContextManager[plt.figure]: return figure to caller to modify and save

    Yields:
        Iterator[ContextManager[plt.figure]]: return figure to caller to modify and save
    """
    fig, ax = PlotUtil.check_ax()
    ax.axis("off")
    fig.text(0.5, 0.5, title, ha="center", va="center")
    yield fig
    plt.close(fig)


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
