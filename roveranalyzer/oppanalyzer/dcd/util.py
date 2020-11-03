import numpy as np
import pandas as pd

from roveranalyzer.uitls import LazyDataFrame


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


def _density_get_raw(csv_path, col_types):
    # todo
    _df = LazyDataFrame.from_path(csv_path)
    _df.dtype = col_types

    select_columns = ["simtime", "x", "y", *list(col_types.keys())]
    df_raw = _df.df(set_index=True, column_selection=select_columns)
    meta = _df.read_meta_data()

    _m = DcdMetaData.from_dict(meta)

    return df_raw, _m


def _apply_real_coords(_df, _meta: DcdMetaData):
    _idxOld = _df.index.to_frame(index=False)
    _idxOld["x"] = _idxOld["x"] * _meta.cell_size
    _idxOld["y"] = _idxOld["y"] * _meta.cell_size
    _idxNew = pd.MultiIndex.from_frame(_idxOld)
    return _df.set_index(_idxNew)


def _full_map(df, _m: DcdMetaData, col_types, real_coords=False):
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


def build_density_map(csv_path, col_types, real_coords=False, full_map=False):
    """
    build density maps from spare csv output.
    expects a csv file with as header simtime;x;y;count;measured_t;received_t.
    The first line must include a metadata line (starting with #) which
    containing CELLSIZE and absolute size of the grid metadata.
    #CELLSIZE=3.000000,DATACOL=-1,IDXCOL=3,SEP=;,XSIZE=581.135000,YSIZE=233.492000
    """
    ret, _m = _density_get_raw(csv_path, col_types)

    if real_coords:
        ret = _apply_real_coords(ret, _m)

    # create full index with missing cells. Values will be set to '0' (type dependent)
    if full_map:
        ret = _full_map(ret, _m, col_types, real_coords)

    return _m, ret


# todo: merge multiple nodes.
def build_local_density_map(csv_path, real_coords=False, full_map=False):
    col = {
        "count": np.int,
        "measured_t": np.float,
        "received_t": np.float,
        "source": np.str,
        "own_cell": np.int,
    }
    meta, df = build_density_map(csv_path, col, real_coords, full_map)
    return meta, df


def build_global_density_map(
    csv_path, real_coords=False, with_id_list=False, full_map=False
):
    col = {
        "count": np.int,
        "measured_t": np.float,
        "received_t": np.float,
        "source": np.str,
        "own_cell": np.int,
    }
    if with_id_list:
        col["node_id"] = np.str
    meta, df = build_density_map(csv_path, col, real_coords, full_map)
    return meta, df
