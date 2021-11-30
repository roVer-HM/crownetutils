from __future__ import annotations

from typing import Tuple, Union

import folium
import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from pandas.core.frame import DataFrame

from roveranalyzer.simulators.opp.provider.hdf.DcDGlobalPosition import (
    DcdGlobalDensity,
    DcdGlobalPosition,
)


class _folium:
    GOOGLE_TILES = "http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga"

    def add_google_tile(self, map: folium.Map) -> _folium:
        folium.TileLayer(
            tiles=self.GOOGLE_TILES, attr="Google", name="Goolge Maps", max_zoom=22
        ).add_to(map)
        return self

    def add_control(self, map: folium.Map) -> _folium:
        folium.LayerControl().add_to(map)
        return self


class _DensityMap:

    folium = _folium()

    def get_annotated_global_map(
        self,
        global_map: Union[DcdGlobalDensity, GeoDataFrame],
        position: Union[DcdGlobalPosition, DataFrame, GeoDataFrame],
        crs: str,
        slice_: slice = slice(None),
    ) -> GeoDataFrame:

        pos = DcdGlobalPosition.as_geo(position, crs, slice_)
        pos = pos.reset_index().groupby(["simtime", "x", "y"])["node_id"].apply(list)
        pos = pos.rename("occupancy")

        if type(global_map) == DcdGlobalDensity:
            map = global_map.geo(crs)[slice_]
        else:
            map = global_map

        map = pd.concat([map, pos], axis=1)

        return map

    def get_interactive(
        self,
        cells: GeoDataFrame = None,
        nodes: GeoDataFrame = None,
        time: Union[None, float] = None,
    ) -> folium.Map:
        map = None
        if cells is not None:
            if time is not None and cells.index.names[0] == "simtime":
                _cells = cells.loc[time]
            else:
                _cells = cells

            _cells = _cells.copy(deep=True).reset_index()
            map = _cells.explore(
                style_kwds={"color": "red", "fill": False},
                name="Cells",
            )
        if map is None and nodes is not None:
            if time is not None and "time" in nodes.columns:
                _nodes = nodes[nodes["time"] == time]
            else:
                _nodes = nodes

            _nodes = _nodes.copy(deep=True).reset_index()
            map = _nodes.explore(
                style_kwds={"color": "blue", "fill": True},
                name="Node Position",
            )
        elif map is not None and nodes is not None:
            if time is not None and "time" in nodes.columns:
                _nodes = nodes[nodes["time"] == time]
            else:
                _nodes = nodes

            _nodes = _nodes.copy(deep=True).reset_index()
            map = _nodes.explore(
                m=map,
                style_kwds={"color": "blue", "fill": True},
                name="Node Position",
            )
        else:
            raise ValueError("At least one input must not be None")

        folium.TileLayer(
            tiles="Stamen Toner", control="True", name="Stamen Toner", max_zoom=22
        ).add_to(map)
        self.folium.add_google_tile(map)
        folium.LayerControl().add_to(map)
        return map


DensityMap = _DensityMap()
