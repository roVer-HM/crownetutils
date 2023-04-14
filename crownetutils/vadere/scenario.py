from __future__ import annotations

import json as j
from re import I

import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon


class Scenario:
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
    def crs(self):
        return self._scenario_json["scenario"]["topography"]["attributes"][
            "referenceCoordinateSystem"
        ]

    @property
    def bound_dict(self):
        return self._scenario_json["scenario"]["topography"]["attributes"]["bounds"]

    @property
    def bound(self):
        b = self._scenario_json["scenario"]["topography"]["attributes"]["bounds"]
        return np.array[
            [b["x"], b["y"]],
            [b["x"], b["y"] + b["width"]],
            [b["x"] + b["height"], b["y"] + b["width"]],
            [b["x"] + b["height"], b["y"]],
        ]

    @property
    def offset(self):
        crs = self.crs
        if "translation" in crs:
            return np.array([crs["translation"]["x"], crs["translation"]["y"]])
        else:
            return np.array([0, 0])

    @property
    def epsg(self):
        return self.crs["epsgCode"]

    @property
    def obstacles(self):
        return self._scenario_json["scenario"]["topography"]["obstacles"]

    @property
    def measurementAreas(self):
        return self._scenario_json["scenario"]["topography"]["measurementAreas"]

    @property
    def stairs(self):
        return self._scenario_json["scenario"]["topography"]["stairs"]

    @property
    def targets(self):
        return self._scenario_json["scenario"]["topography"]["targets"]

    @property
    def target_changers(self):
        return self._scenario_json["scenario"]["topography"]["targetChangers"]

    @property
    def absorbing_areas(self):
        return self._scenario_json["scenario"]["topography"]["absorbingAreas"]

    @property
    def sources(self):
        return self._scenario_json["scenario"]["topography"]["sources"]

    @property
    def dynamic_elements(self):
        return self._scenario_json["scenario"]["topography"]["dynamicElements"]

    @property
    def attr_pedestrian(self):
        return self._scenario_json["scenario"]["topography"]["attributesPedestrian"]

    @property
    def bound(self):
        return self.topography["attributes"]["bounds"]

    @staticmethod
    def shape_to_list(shape, to_shapely: bool = False):
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
        if to_shapely:
            if all(points[0] == points[-1]):
                return Polygon(points)
            else:
                return Polygon(np.append(points, points[0]).reshape((-1, 2)))

        else:
            return points

    def topography_frame(self, to_crs: str | None = None):
        default_colors = {
            "obstacles": "#B3B3B3",  # grey
            "targets": "#DD8452",  # orange
            "sources": "#55A868",  # green
        }

        data = []
        for element, color in default_colors.items():
            elements = self.topography[element]
            for e in elements:
                polygon = self.shape_to_list(e["shape"], to_shapely=True)
                style = dict(
                    fillColor=color,
                    fillOpacity=1.0,
                    weight=0,
                    zIndex=100,
                    color="#000000",
                )
                info = dict(e)
                del info["shape"]
                data.append((element, color, style, info, polygon))

        df = gpd.GeoDataFrame(
            data, columns=["type", "fillColor", "style", "info", "geometry"]
        )
        if to_crs is not None:
            df["geometry"] = df["geometry"].translate(
                xoff=self.offset[0], yoff=self.offset[1], zoff=0.0
            )

            df.crs = self.epsg
            df = df.to_crs(epsg=to_crs.replace("EPSG:", ""))

        return df
