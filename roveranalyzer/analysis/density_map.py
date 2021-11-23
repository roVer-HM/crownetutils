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


class _DensityMap:

    GOOGLE_TILES = "http://mt0.google.com/vt/lyrs=m&hl=en&x={x}&y={y}&z={z}&s=Ga"

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
    ) -> folium.Map:
        map = None
        if nodes is not None:
            map = cells.explore(
                color="blue",
                name="Node Position",
            )
        if map is None and nodes is not None:
            map = nodes.explore(
                color="red",
                name="Cells",
            )
        elif map is not None and nodes is not None:
            map = nodes.explore(m=map, color="red", name="Cells")
        else:
            raise ValueError("At least one input must not be None")

        folium.TileLayer(
            tiles="Stamen Toner", control="True", name="Stamen Toner", max_zoom=22
        ).add_to(map)
        folium.TileLayer(
            tiles=self.GOOGLE_TILES, attr="Google", name="Goolge Maps", max_zoom=22
        ).add_to(map)
        folium.LayerControl().add_to(map)
        return map


DensityMap = _DensityMap()
