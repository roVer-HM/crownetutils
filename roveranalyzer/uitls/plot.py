import matplotlib.pyplot as plt
import pandas as pd


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
