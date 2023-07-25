from importlib.resources import read_text

__version__ = read_text("crownetutils", "version.txt").strip()
import matplotlib

from crownetutils.utils.colors import _register_cmap

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
_register_cmap()
