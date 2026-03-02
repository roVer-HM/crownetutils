import logging
import shutil
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
    if plt.rcParams["text.usetex"]:
        if not shutil.which("latex"):
            logging.warning("tex/latex not installed set `text.usetex: False`")
            plt.rcParams.update({"text.usetex": False})

    return old


def style_context(style):
    if isinstance(style, tuple):
        with resource_path(*style) as path:
            return plt.style.context(path)
    else:
        raise ValueError()
