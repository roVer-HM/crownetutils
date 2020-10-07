import json as j

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches


class VaderScenarioPlotHelper:
    """
    Plot helper that adds patches for obstacles, targets and source to a given
    axis.
    """

    default_colors = {
        "obstacles": "#B3B3B3",  # grey
        "targets": "#DD8452",  # orange
        "sources": "#55A868",  # green
    }

    def __init__(self, path):
        with open(path, "r", encoding="utf-8") as f:
            self._scenario_json = j.load(f)

    @property
    def scenario(self):
        return self._scenario_json["scenario"]

    @property
    def topography(self):
        return self._scenario_json["scenario"]["topography"]

    @property
    def bound(self):
        return self.topography["attributes"]["bounds"]

    @staticmethod
    def shape_to_list(shape):
        if shape["type"] == "POLYGON":
            points = np.array([[p["x"], p["y"]] for p in shape["points"]])
        elif shape["type"] == "RECTANGLE":
            start = np.array([shape["x"], shape["y"]])
            points = start
            points = np.append(points, start + np.array([shape["width"], 0]), axis=0)
            points = np.append(
                points, start + np.array([shape["width"], shape["height"]]), axis=0
            )
            points = np.append(points, start + np.array([0, shape["height"]]), axis=0)
            points = points.reshape((-1, 2))
        else:
            raise ValueError("Expected POLYGON or RECTANGLE")
        return points

    def add_obstacles(self, ax: plt.Axes):
        return self.add_patches(ax, {"obstacles": "#B3B3B3"})

    def add_patches(self, ax: plt.Axes, element_type_map: dict = None):

        if element_type_map is None:
            element_type_map = self.default_colors

        for element, color in element_type_map.items():
            elements = self.topography[element]
            polygons = [self.shape_to_list(e["shape"]) for e in elements]
            for poly in polygons:
                patch = patches.Polygon(poly, facecolor=color, fill=True, closed=True)
                ax.add_patch(patch)

        return ax
