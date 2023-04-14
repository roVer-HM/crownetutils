from copy import deepcopy
from importlib.resources import path as resource_path

import matplotlib
import matplotlib.pyplot as plt

STYLE_SIMPLE_169 = ("crownetutils.utils.styles", "simple16x9.mplstyle")
STYLE_TEX = ("crownetutils.utils.styles", "default_paper_tex.mplstyle")


def load_matplotlib_style(style) -> matplotlib.RcParams:
    old = deepcopy(plt.rcParams)
    if isinstance(style, tuple):
        with resource_path(*style) as style_path:
            plt.rcParams.update(plt.rcParamsDefault)
            plt.style.use(style_path)
    else:
        plt.rcParams.update(plt.rcParamsDefault)
        plt.style.use(style)
    return old
