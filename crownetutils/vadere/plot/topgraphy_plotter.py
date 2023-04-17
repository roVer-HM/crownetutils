from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches
from shapely.geometry import Polygon

from crownetutils.vadere.scenario import Scenario


class VadereTopographyPlotter:
    """
    Plot helper that adds patches for obstacles, targets and source to a given
    axis.
    """

    default_colors = {
        "obstacles": "#B3B3B3",  # grey
        "targets": "#DD8452",  # orange
        "sources": "#55A868",  # green
    }

    def __init__(self, scenario: Union[str, Scenario]):
        if type(scenario) == Scenario:
            self.scenario = scenario
        else:
            self.scenario = Scenario(scenario)

    def add_obstacles(self, ax: plt.Axes):
        return self.add_patches(ax, {"obstacles": "#B3B3B3"})

    def add_patches(self, ax: plt.Axes, element_type_map: dict = None, bound=None):
        if element_type_map is None:
            element_type_map = self.default_colors

        if bound is not None:
            x, y, w, h = bound
            bound: Polygon = Polygon(
                [(x, y), (x + w, y), (x + w, y + h), (x, y + h), (x, y)]
            )

        for element, color in element_type_map.items():
            elements = self.scenario.topography[element]
            polygons = [
                self.scenario.shape_to_list(e["shape"], to_shapely=True)
                for e in elements
            ]
            for poly in polygons:
                if bound is not None:
                    if poly.intersects(bound):
                        poly = poly.intersection(bound)
                # poly is closed patches does not have to close it,
                patch = patches.Polygon(
                    list(poly.exterior.coords),
                    edgecolor=color,
                    facecolor=color,
                    fill=True,
                    closed=False,
                )

                ax.add_patch(patch)

        return ax

    def get_legal_cells(self, xy_slices: Tuple[slice, slice], c=5.0):
        """Filter cells based on __FULL__ encloser within obstacles.
        Args:
            xy_slices (Tuple[slice, slice]): area to search of overlapping with obstacles.
            c (float, optional): Size of cells. (Step in slices will be ignored!) . Defaults to 5.0.
        Returns:
            Tuple[pd.MultiIndex, pd.MultiIndex]: Free and covered cells described as pd.MultiIndex.
        """

        _covered = []
        _free = []
        _x, _y = xy_slices
        obs: List[Polygon] = [
            self.scenario.shape_to_list(s["shape"], to_shapely=True)
            for s in self.scenario.obstacles
        ]
        for x in np.arange(_x.start, _x.stop, c):
            for y in np.arange(_y.start, _y.stop, c):
                idx = (x, y)
                cell = Polygon([[x, y], [x + c, y], [x + c, y + c], [x, y + c], [x, y]])
                for _o in obs:
                    if _o.covers(cell):
                        _covered.append(idx)
                        break
                if _covered[-1] != idx:
                    _free.append(idx)

        _covered = pd.MultiIndex.from_tuples(_covered, names=["x", "y"])
        _free = pd.MultiIndex.from_tuples(_free, names=["x", "y"])

        return _free, _covered
