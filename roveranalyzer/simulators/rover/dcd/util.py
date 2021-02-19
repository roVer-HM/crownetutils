from itertools import repeat

import numpy as np
import pandas as pd

from roveranalyzer.utils import LazyDataFrame


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
        return cls(cell_size, cell_count, bound, node_id)

    def __init__(self, cell_size, cell_count, bound, node_id):
        self.cell_size = cell_size
        self.cell_count = cell_count
        self.bound = bound
        self.node_id = node_id

    def is_global(self):
        return self.node_id == "global"

    def is_node(self):
        return self.node_id not in ["global", "all"]

    def is_all(self):
        return self.node_id == "all"

    def is_same(self, other):
        return self.cell_size == other.cell_size and self.bound == other.bound

    def grid_index_2d(self, real_coords=False):
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


def _density_get_raw(csv_path, index, col_types):
    """
    read csv and set index
    """
    _df = LazyDataFrame.from_path(csv_path)
    _df.dtype = col_types

    select_columns = [*index, *list(col_types.keys())]
    df_raw: pd.DataFrame = _df.df(set_index=False, column_selection=select_columns)
    df_raw = df_raw.set_index(index)
    df_raw = df_raw.sort_index()

    meta = _df.read_meta_data()
    _m = DcdMetaData.from_dict(meta)

    return df_raw, _m


def _apply_real_coords(_df, _meta: DcdMetaData):
    _idxOld = _df.index.to_frame(index=False)
    _idxOld["x"] = _idxOld["x"] * _meta.cell_size
    _idxOld["y"] = _idxOld["y"] * _meta.cell_size
    _idxNew = pd.MultiIndex.from_frame(_idxOld)
    return _df.set_index(_idxNew)


# deprecated
def _full_map(df, _m: DcdMetaData, index, col_types, real_coords=False):
    """
    create full index: time * numXCells * numYCells
    """
    idx = _m.create_full_index_from_df(df, real_coords)
    # create zero filled data frame with index
    expected_columns = list(col_types.keys())
    ret = pd.DataFrame(
        data=np.zeros((len(idx), len(expected_columns))), columns=expected_columns
    )
    # set index and update with raw measures. (most will stay at zero)
    ret = ret.set_index(idx)
    ret.update(df)
    ret = ret.astype(df.dtypes)
    return ret


def run_pool(pool, fn, kwargs_iter):
    starmap_args = zip(repeat(fn), kwargs_iter)
    return pool.starmap(apply_pool_kwargs, starmap_args)


def apply_pool_kwargs(fn, kwargs):
    return fn(**kwargs)


def build_density_map(
    csv_path,
    index,
    column_types,
    real_coords=False,
    add_missing_cells=False,
    df_filter=None,
):
    """
    build density maps from spare csv output.
    expects a csv file with as header simtime;x;y;count;measured_t;received_t.
    The first line must include a metadata line (starting with #) which
    containing CELLSIZE and absolute size of the grid metadata.
    #CELLSIZE=3.000000,DATACOL=-1,IDXCOL=3,SEP=;,XSIZE=581.135000,YSIZE=233.492000
    """
    print(f"load {csv_path}")
    ret, meta = _density_get_raw(csv_path, index, column_types)

    if df_filter is not None:
        # apply early filter to remove not needed data to increase performance
        ret = df_filter(ret)

    if real_coords:
        ret = _apply_real_coords(ret, meta)

    # create full index with missing cells. Values will be set to '0' (type dependent)
    if add_missing_cells:
        ret = _full_map(ret, meta, index, column_types, real_coords)

    return meta, ret
