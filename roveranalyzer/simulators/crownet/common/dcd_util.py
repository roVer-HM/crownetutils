from itertools import repeat
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
from geopandas import GeoDataFrame
from pandas import IndexSlice as Idx
from shapely import geometry
from shapely.geometry import Point, box

from roveranalyzer.simulators.crownet.common.dcd_metadata import DcdMetaData
from roveranalyzer.utils.dataframe import LazyDataFrame
from roveranalyzer.utils.misc import Timer


def create_error_df(map_df, glb_df):
    """
    Extract count errors for each count measure based on cell(x, y), time and owner(Id).
    RowIndex('simtime', 'x', 'y', 'ID')
    ColumnIndex('count', 'err', 'owner_dist', 'sqerr')
    """
    t = Timer.create_and_start("create_error_df", label="create_error_df")
    # copy count information of maps and ground truth into one data frame
    d = pd.DataFrame(map_df.loc[:, ["count", "x_owner", "y_owner"]].copy())
    g: pd.DataFrame = glb_df.copy()
    g["ID"] = 0
    g["x_owner"] = glb_df.index.get_level_values("x")
    g["y_owner"] = glb_df.index.get_level_values("y")
    g = g.reset_index().set_index(["simtime", "x", "y", "ID"], drop=True)
    all = pd.concat([g, d])

    # explode missing count values of node data based on global data. Due to the fact that the ground truth (ID=0) is
    # present in the data frame, the pivot_table will add missing 'count' values of nodes. Missing values are filled
    # with '0' as nothing was counted. Wrong counts of nodes will lead to '0' filled values in the ground truth
    # column (ID=0)
    all_pivot = all.pivot_table(
        values=["count", "x_owner", "y_owner"],
        index=["simtime", "x", "y"],
        columns=["ID"],
        aggfunc=sum,
        fill_value=0,  # add NaN later for values where node has left the simulation
    )
    # rename count index to values as new columns will be added for 'err' and 'sqerr' to this index
    all_pivot.columns = all_pivot.columns.rename("values", level=0)
    # swap index levels for convinces
    all_pivot.columns = all_pivot.columns.swaplevel()

    # calculate count error 'err' (negative: underestimated / positive: overestimated) and 'sqerr' (squared err)
    all_times = all_pivot.index.get_level_values("simtime").unique().to_numpy()
    all_ids = all_pivot.columns.get_level_values("ID").unique().to_numpy()
    _truth = all_pivot.loc[:, Idx[0, "count"]].copy()
    _x_idx = all_pivot.index.get_level_values("x")
    _y_idx = all_pivot.index.get_level_values("y")
    for _id in all_ids:
        all_pivot[_id, "err"] = all_pivot.loc[Idx[:], Idx[_id, "count"]] - _truth
        all_pivot[_id, "sqerr"] = all_pivot[_id, "err"] ** 2
        all_pivot[_id, "owner_dist"] = np.sqrt(
            (all_pivot[_id, "x_owner"] - _x_idx) ** 2
            + (all_pivot[_id, "y_owner"] - _y_idx) ** 2
        )
        # clear values for times where node _id is not present
        if _id > 0:
            present_at_times = (
                map_df.loc[Idx[:, :, :, :, _id]]
                .index.get_level_values("simtime")
                .unique()
                .to_numpy()
            )
            not_present_at_times = np.setdiff1d(all_times, present_at_times)
            all_pivot.loc[Idx[not_present_at_times, :], Idx[_id]] = np.nan

    # set owner_dist for ground truth to 0
    all_pivot[0, "owner_dist"] = 0

    # remove x_owner / y_owner columns
    r_idx = np.array(
        [
            all_ids,
            np.full(all_ids.shape, "x_owner"),
            all_ids,
            np.full(all_ids.shape, "y_owner"),
        ]
    ).T
    r_idx = np.concatenate((r_idx[:, :2], r_idx[:, 2:]))
    r_idx = [tuple([int(i[0]), i[1]]) for i in r_idx]
    all_pivot = all_pivot.drop(r_idx, axis=1)

    # sort columns for convinces
    all_pivot = all_pivot.sort_index(axis="columns")
    t.stop()
    # remove pivot property and flat the table
    all_pivot = all_pivot.stack(0)
    return all_pivot


def delay_feature(_df_ret, **kwargs):
    # calculate features (delay, AoI_NR (measured-to-now), AoI_MNr (received-to-now)
    now = _df_ret.index.get_level_values("simtime")
    _df_ret["delay"] = _df_ret["received_t"] - _df_ret["measured_t"]
    # AoI_NM == measurement_age (time since last measurement)
    _df_ret["measurement_age"] = now - _df_ret["measured_t"]
    # AoI_NR == update_age (time since last update)
    _df_ret["update_age"] = now - _df_ret["received_t"]
    return _df_ret


def owner_dist_feature(_df_ret, meta: DcdMetaData, **kwargs):
    if "global_position" not in kwargs:
        # Warning: may lead to error if not all owner locations are part of the data frame
        return owner_dist_feature_old(_df_ret, **kwargs)

    if "global_metadata" in kwargs and kwargs["global_metadata"].is_entropy_data():
        # entropy based map. Owner distance invalid!
        _df_ret["x_owner"] = -1.0
        _df_ret["y_owner"] = -1.0
        _df_ret["owner_dist"] = -1.0
        return _df_ret

    # access global position map and extract positions of the current node.
    glb_pos = kwargs["global_position"]
    # node_id = _df_ret.index.get_level_values("ID").unique()[0]
    node_id = int(meta.node_id)
    node_positions = glb_pos.loc[Idx[:, node_id], :]  # .reset_index()
    node_positions = node_positions.rename(columns={"x": "x_owner", "y": "y_owner"})
    node_positions.index.set_names("ID", level=1, inplace=True)

    index_names = _df_ret.index.names
    # merge extracted node_positions with _df_ret
    _df_ret = pd.merge(
        _df_ret.reset_index(),
        node_positions,
        on=["ID", "simtime"],
        how="left",
    ).reset_index(drop=True)

    # compute distance between cells (row) and the owner's location at the given time
    _df_ret["owner_dist"] = np.sqrt(
        (_df_ret["x"] - _df_ret["x_owner"]) ** 2
        + (_df_ret["y"] - _df_ret["y_owner"]) ** 2
    )
    _df_ret = _df_ret.set_index(index_names, drop=True, verify_integrity=True)

    return _df_ret


def owner_dist_feature_old(_df_ret, **kwargs):
    """
    Assume each node logs its own position in the DCD map thus the location
    of each node can be extracted for each time stamp solely from the data frame
    """
    index_names = _df_ret.index.names

    # Distance to owner location.
    # get owner positions for each time step {ID/simtime}[x_owner,y_owner]
    # fixme knowledge about "source" in index does not belong here
    if "source" in index_names:
        owner_locations = (
            _df_ret.loc[_df_ret["own_cell"] == 1, []]
            .index.to_frame(index=False)
            .drop(columns=["source"])
            .drop_duplicates()
            .set_index(["ID", "simtime"], drop=True, verify_integrity=True)
            .rename(columns={"x": "x_owner", "y": "y_owner"})
        )
    else:
        owner_locations = (
            _df_ret.loc[_df_ret["own_cell"] == 1, []]
            .index.to_frame(index=False)
            .drop_duplicates()
            .set_index(["ID", "simtime"], drop=True, verify_integrity=True)
            .rename(columns={"x": "x_owner", "y": "y_owner"})
        )

    # merge owner position
    _df_ret = pd.merge(
        _df_ret.reset_index(),
        owner_locations,
        on=["ID", "simtime"],
    ).reset_index(drop=True)

    _df_ret["owner_dist"] = np.sqrt(
        (_df_ret["x"] - _df_ret["x_owner"]) ** 2
        + (_df_ret["y"] - _df_ret["y_owner"]) ** 2
    )
    _df_ret = _df_ret.set_index(index_names, drop=True, verify_integrity=True)

    return _df_ret


def check_has_own_cell(_df):
    """
    Ensure at ;east
    """
    return _df.loc[_df["own_cell"] == 0, []].shape[0] > 0


def read_csv(
    csv_path,
    _index_types: dict,
    _col_types: dict,
    real_coords=True,
    df_filter=None,
) -> Tuple[pd.DataFrame, DcdMetaData]:
    """
    read csv and set index
    """
    _df = LazyDataFrame.from_path(csv_path)
    _df.dtype = {**_index_types, **_col_types}
    select_columns = list(_df.dtype.keys())
    index_names = list(_index_types.keys())
    df_raw: pd.DataFrame = _df.df(set_index=False, column_selection=select_columns)
    df_raw = df_raw.set_index(index_names)
    df_raw = df_raw.sort_index()

    # apply given filters first (if any)
    if df_filter is not None:
        if type(df_filter) == list:
            for _f in df_filter:
                df_raw = _f(df_raw)
        else:
            # apply early filter to remove not needed data to increase performance
            df_raw = df_filter(df_raw)

    meta = _df.read_meta_data()
    _m = DcdMetaData.from_dict(meta)

    if real_coords:
        df_raw = _apply_real_coords(df_raw, _m)

    return df_raw, _m


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


def remove_not_selected_cells(df: pd.DataFrame):
    """
    remove all rows not selected in the density map.
    """
    return df.loc[df["selection"].notna()].copy(deep=True)


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
        if type(df_filter) == list:
            for _f in df_filter:
                ret = _f(ret)
        else:
            # apply early filter to remove not needed data to increase performance
            ret = df_filter(ret)

    if real_coords:
        ret = _apply_real_coords(ret, meta)

    return meta, ret
