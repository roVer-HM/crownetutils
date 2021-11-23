from typing import Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from geopandas import GeoDataFrame
from pandas.core.frame import DataFrame

from roveranalyzer.simulators.opp.provider.hdf.DcDGlobalPosition import (
    DcdGlobalDensity,
    DcdGlobalPosition,
)


class _DensityMap:
    def get_annotated_global_map(
        self,
        global_map: Union[DcdGlobalDensity, GeoDataFrame],
        position: Union[DcdGlobalPosition, DataFrame, GeoDataFrame],
        crs: str,
        slice_: slice = slice(None),
    ) -> GeoDataFrame:

        pos = DcdGlobalPosition.as_geo(position, crs, slice_)
        pos = pos.reset_index().groupby(["simtime", "x", "y"])["node_id"].apply(list)

        if type(global_map) == DcdGlobalDensity:
            map = global_map.geo(crs)[slice_]
        else:
            map = global_map

        map = pd.concat([map, pos], axis=1)

        return map


DensityMap = _DensityMap()
