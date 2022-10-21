from typing import Tuple, Union

import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import IndexSlice as Idx
from shapely.geometry import Point, box

from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import ProviderVersion


class DcdMetaData:
    expected_keys = ["XSIZE", "YSIZE", "CELLSIZE", "NODE_ID"]

    @classmethod
    def from_dict(cls, meta):
        # check if all metadata keys are present
        if not all([k in meta for k in cls.expected_keys]):
            raise ValueError(
                f"expected all metadata keys present. "
                f"expected: {', '.join(cls.expected_keys)} | "
                f"found: {', '.join(meta.keys())}"
            )
        # bound of szenario
        bound = [float(meta["XSIZE"]), float(meta["YSIZE"])]
        # cell size. First cell with [0, 0] is lower left cell
        cell_size = float(meta["CELLSIZE"])
        cell_count = [int(bound[0] / cell_size + 1), int(bound[1] / cell_size + 1)]
        node_id = meta["NODE_ID"]
        _meta = cls(
            cell_size,
            cell_count,
            bound,
            node_id,
            map_type=meta.get("MAP_TYPE", ""),
            version=meta.get("VERSION", "0.1"),
            data_type=meta.get("DATATYPE", "pedestrianCount"),
        )
        if all([k in meta for k in ["XOFFSET", "YOFFSET"]]):
            _meta.offset = [float(meta["XOFFSET"]), float(meta["YOFFSET"])]
        return _meta

    def __init__(
        self,
        cell_size,
        cell_count,
        bound,
        node_id=0,
        offset=[0.0, 0.0],
        epsg="",
        map_extend_x=[0, 0],
        map_extend_y=[0, 0],
        map_type="",
        version="0.1",
        data_type="pedestrianCount",
    ):
        self.cell_size = cell_size
        self.cell_count = cell_count
        self.bound = bound
        self.node_id = node_id
        self.offset = offset
        self.epsg = epsg
        self.map_extend_x = map_extend_x
        self.map_extend_y = map_extend_y
        self.map_type = map_type
        self.version = ProviderVersion(version)
        self.data_type = data_type

    def is_global(self):
        return self.node_id == "global" or self.node_id == -1

    def is_node(self):
        return self.node_id not in ["global", "all"]

    def mgrid(self):
        return np.mgrid[
            slice(0, self.bound[0] + self.cell_size, self.cell_size),
            slice(0, self.bound[1] + self.cell_size, self.cell_size),
        ]

    def is_all(self):
        return self.node_id == "all"

    def is_entropy_data(self):
        return "entropy" in self.map_type

    def is_same(self, other):
        return self.cell_size == other.cell_size and self.bound == other.bound

    def bound_gdf(self, crs: Union[str, None] = None) -> GeoDataFrame:
        df = pd.DataFrame([["bound"]], columns=["Name"])
        _o = np.abs(self.offset)

        x_size = self.map_extend_x[1] - self.map_extend_x[0]
        y_size = self.map_extend_y[1] - self.map_extend_y[0]
        geometry = [
            box(
                _o[0] + self.map_extend_x[0],
                _o[1] + self.map_extend_y[0],
                _o[0] + x_size,
                _o[1] + y_size,
            )
        ]
        _df = GeoDataFrame(df, geometry=geometry, crs=self.epsg)
        if crs is not None:
            _df = _df.to_crs(epsg=crs.replace("EPSG:", ""))
        return _df

    def grid_index_2d(self, real_coords=False) -> pd.MultiIndex:
        """
        crate full (cartesian) index based on map dimension.
        If real_coords is set use lower left corner of cell as value
        """
        if real_coords:
            _idx = [
                np.arange(self.x_count) * self.cell_size,  # numXCell
                np.arange(self.y_count) * self.cell_size,  # numYCell
            ]
        else:
            _idx = [
                np.arange(self.x_count),  # numXCell
                np.arange(self.y_count),  # numYCell
            ]
        return pd.MultiIndex.from_product(_idx, names=("x", "y"))

    def create_full_index(self, times, real_coords=False):
        if real_coords:
            _idx = [
                times,  # time
                np.arange(self.x_count) * self.cell_size,  # numXCell
                np.arange(self.y_count) * self.cell_size,  # numYCell
            ]
        else:
            _idx = [
                times,  # time
                np.arange(self.x_count),  # numXCell
                np.arange(self.y_count),  # numYCell
            ]
        return pd.MultiIndex.from_product(_idx, names=("simtime", "x", "y"))

    def create_min_grid_index(
        self, map_idx: pd.MultiIndex, difference_only: bool = True
    ):
        x_min = map_idx.get_level_values("x").min()
        x_max = map_idx.get_level_values("x").max()
        y_min = map_idx.get_level_values("y").min()
        y_max = map_idx.get_level_values("y").max()
        _idx = pd.MultiIndex.from_product(
            [
                np.arange(x_min, x_max + self.cell_size, self.cell_size),
                np.arange(y_min, y_max + self.cell_size, self.cell_size),
            ],
            names=map_idx.names,
        )
        if difference_only:
            return _idx.difference(map_idx)
        else:
            return _idx

    def empty_df(self, value_name, real_coords=True):
        """
        Crate an empty dataframe containing all cells of the map.
        Returns df with shape(N,1), and index [x, y]
        """
        full_index = self.grid_index_2d(real_coords)
        df = pd.DataFrame(np.zeros((len(full_index), 1)), columns=[value_name])
        df = df.set_index(full_index)
        return df

    def update_missing(self, df, real_coords=True):
        """
        Creates
        """
        index_names = list(df.index.names)
        if index_names == ["x", "y"]:
            full_index = self.grid_index_2d(real_coords)
        else:
            wrong_index = ",".join([f"'{i}'" for i in index_names])
            raise ValueError(
                f"Unsupported index. Expected ['x', 'y'] but got [{wrong_index}]"
            )

        columns = list(df.columns)
        full_df = pd.DataFrame(
            np.zeros((len(full_index), len(columns))), columns=columns
        )
        full_df = full_df.set_index(full_index)
        full_df.update(df)
        return full_df

    def create_full_index_from_df(self, df, real_coords=False):
        return self.create_full_index(
            df.index.levels[0].to_numpy(dtype=float), real_coords
        )

    @property
    def x_dim(self):
        return self.bound[0]

    @property
    def X(self):
        return np.arange(self.x_count * self.cell_size, step=self.cell_size)

    @property
    def Y(self):
        return np.arange(self.y_count * self.cell_size, step=self.cell_size)

    @property
    def X_flat(self):
        """
        add one additional element at the end to use flat shading of matplotlib without dropping
        data. See. https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html?highlight=pcolormesh%20grids%20shading#centered-coordinates
        """
        return np.arange((1 + self.x_count) * self.cell_size, step=self.cell_size)

    @property
    def Y_flat(self):
        """
        add one additional element at the end to use flat shading of matplotlib without dropping
        data. See. https://matplotlib.org/gallery/images_contours_and_fields/pcolormesh_levels.html?highlight=pcolormesh%20grids%20shading#centered-coordinates
        """
        return np.arange((1 + self.y_count) * self.cell_size, step=self.cell_size)

    @property
    def y_dim(self):
        return self.bound[1]

    @property
    def x_count(self):
        return self.cell_count[0]

    @property
    def y_count(self):
        return self.cell_count[1]
