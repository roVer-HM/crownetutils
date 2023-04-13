from __future__ import annotations

import collections
import json
from copy import deepcopy
from os.path import join, split
from re import I

import numpy as np
import pandas as pd
from omnetinireader.config_parser import ObjectValue

from roveranalyzer.analysis.common import Simulation
from roveranalyzer.analysis.dpmm.hdf.dpmm_count_provider import DpmmCount
from roveranalyzer.analysis.dpmm.hdf.dpmm_provider import DpmmProvider
from roveranalyzer.analysis.flaskapp.application.utils import threaded_lru
from roveranalyzer.omnetpp.scave import CrownetSql
from roveranalyzer.utils.dataframe import LazyDataFrame
from roveranalyzer.utils.logging import timing
from roveranalyzer.utils.misc import Project
from roveranalyzer.vadere.scenario import Scenario


@threaded_lru(maxsize=16)
@timing
def get_count_index(sim: Simulation) -> pd.DataFrame:
    p = sim.builder.count_p
    with p.query as ctx:
        count_index: pd.DataFrame = ctx.select(key=p.group, columns=[])
    return count_index
    # return count_index.index.to_frame().reset_index(drop=True)


@timing
def get_node_index(sim: Simulation):
    count_index = get_count_index(sim)
    return count_index.index.get_level_values("ID").unique().sort_values()


@timing
def get_time_index(sim: Simulation):
    count_index = get_count_index(sim)
    return count_index.index.get_level_values("simtime").unique().sort_values()


@threaded_lru(maxsize=16)
@timing
def get_cells(sim: Simulation):
    count_index = get_count_index(sim)
    return count_index.index.droplevel(["simtime", "ID"]).unique().sort_values()


@threaded_lru(maxsize=16)
@timing
def get_host_ids(sim: Simulation):
    host_ids = sim.sql.host_ids()
    host_ids[0] = "Global View (Ground Truth)"
    host_ids = collections.OrderedDict(sorted(host_ids.items()))
    return host_ids


@threaded_lru(maxsize=64)
@timing
def get_topography_json(sim: Simulation):
    _s = sim.sql.vadere_scenario
    if _s is None:
        print("no Vadere scenario file found.")
        return {}
    name = split(_s)[-1]
    s = Scenario(join(sim.data_root, f"vadere.d/{name}"))
    df = s.topography_frame(to_crs=Project.WSG84_lat_lon)
    return json.loads(df.reset_index().to_json())


@threaded_lru(maxsize=64)
def get_node_tile_geojson_for(sim: Simulation, time_value, node_value):
    with sim.pos.query as ctx:
        nodes = ctx.select(
            "trajectories",
            where=f"(time <= {time_value}) & (time > {time_value - 0.4})",
        )
    nodes = sim.sql.apply_geo_position(nodes, Project.UTM_32N, Project.WSG84_lat_lon)
    nodes["tooltip"] = nodes["hostId"].astype(str) + " " + nodes["host"]
    nodes["color"] = "#0000bb"
    nodes["color"] = nodes["color"].where(
        nodes["hostId"] != node_value, other="#bb0000"
    )
    return json.loads(nodes.reset_index().to_json())


@threaded_lru(maxsize=64)
def get_cell_tile_geojson_for(sim: Simulation, time_value, node_id):
    map_data = sim.builder.count_p.geo(Project.WSG84_lat_lon)[
        pd.IndexSlice[time_value, :, :, node_id]
    ]
    j = json.loads(map_data.reset_index().to_json())
    return j


@threaded_lru(maxsize=64)
@timing
def get_erroneous_cells(sim: Simulation):

    with sim.builder.count_p.query as ctx:
        erroneous_cells = ctx.select(
            key=sim.builder.count_p.group,
            where="(ID>0) & (err > 0)",
            columns=["count", "err"],
        )

    _mask = erroneous_cells["count"] == erroneous_cells["err"]
    erroneous_cells = erroneous_cells[_mask]
    _err_ret = (
        erroneous_cells.groupby(by=["x", "y"])
        .sum()
        .sort_values("count", ascending=False)
        .reset_index()
        .set_index(["x", "y"])
        .iloc[0:30]
        .copy()
    )
    _err_ret["node_ids"] = ""
    for g, _ in _err_ret.groupby(["x", "y"]):
        l = (
            erroneous_cells.groupby(by=["x", "y", "ID"])
            .sum()
            .loc[g]
            .nlargest(5, "err")
            .index.to_list()
        )
        _d = ", ".join([str(i) for i in l])
        _err_ret.loc[g, "node_ids"] = _d

    return _err_ret.reset_index()


@threaded_lru(maxsize=64)
@timing
def get_cell_error_data(sim: Simulation, cell_id):
    # todo: for what do i need this ??
    i = pd.IndexSlice
    cell_id = 0 if cell_id is None else cell_id
    _cell = get_cells(sim)[cell_id]
    with sim.builder.count_p.query as ctx:
        # all nodes, all times given _cell
        ca = ctx.select(
            key=sim.builder.count_p.group,
            where=f"(x == {float(_cell[0])}) & (y == {float(_cell[1])})",
        )
    ca = ca.reset_index(["x", "y"]).sort_index()

    if sim.sql.is_entropy_map():
        ca = ca[~ca["missing_value"]]

    # only ground truth for given cell
    ca_0 = ca.loc[i[:, 0], :]

    # create missing [time, ID] index for ground truth (i.e. np.zeros(..))
    time_index = get_time_index(sim)
    idx_diff = pd.Index(zip(time_index, np.zeros(time_index.shape[0]))).difference(
        ca_0.index
    )

    # append missing counts to data frame...
    _add_zero = pd.DataFrame(index=idx_diff, columns=ca_0.columns).fillna(0)
    ca = pd.concat([ca, _add_zero])
    ca = ca.reset_index()
    ca["id"] = ca["ID"].astype("str")
    return ca


@threaded_lru(maxsize=64)
@timing
def get_count_diff(sim: Simulation, node_id: int = -1):
    if node_id == -1:
        # all nodes -> mean count
        return sim.get_dcdMap().count_diff(id_slice=slice(1, None, None))
    else:
        return sim.get_dcdMap().count_diff(id_slice=node_id)


@threaded_lru(maxsize=64)
@timing
def get_node_ids_for_cell(sim: Simulation, cell_id):
    ca = get_cell_error_data(sim, cell_id)
    return ca["ID"].unique()


@threaded_lru(maxsize=16)
@timing
def get_beacon_df(sim: Simulation):

    df = LazyDataFrame.from_path(join(sim.data_root, "beacons.csv"))
    m = df.read_meta_data(default={})
    b = df.df()
    if "version" in m:
        b.user_version = float(m["version"])

    if any(b["cell_x"] > 0x0FFF_FFFF) or any(b["cell_y"] > 0x0FFF_FFFF):
        raise ValueError("cell positions exceed 28 bit.")
    bound = sim.builder.count_p.get_attribute("cell_bound")
    b["posY"] = bound[1] - b["posY"]  # fix translation !
    b["cell_id"] = (b["cell_x"].values << 28) + b["cell_y"]
    return b


def _get_beacon_entry_exit_v0(
    b: pd.DataFrame, sim: Simulation, node_id: int, cell: tuple
):

    c_size = sim.builder.count_p.get_attribute("cell_size")
    beacon_mask = (
        (b["table_owner"] == node_id)
        & (b["posX"] > (cell[0] - c_size))
        & (b["posX"] < (cell[0] + 2 * c_size))
        & (b["posY"] > (cell[1] - c_size))
        & (b["posY"] < (cell[1] + 2 * c_size))
    )
    bs = b[beacon_mask]

    bs = bs.set_index(
        ["table_owner", "source_node", "event_time", "cell_x", "cell_y", "pkt_count"]
    ).sort_index()
    bs["cell_change"] = 0
    bs["cell_change_count"] = 0
    bs["cell_change_cumsum"] = 0
    _m = bs["event"] == "ttl_reached"
    bs.loc[_m, ["cell_change_count"]] = -1
    if bs.empty:
        return bs
    bs_missing = []
    bs_missing_idx = []
    if not bs.index.is_unique:
        raise RuntimeError(
            f"Beacons csv is not unique for node {node_id} and cell {cell}"
        )

    for g, df in bs.groupby(by=["table_owner", "source_node"]):
        # find rows where a changes  between cells happens (i.e. cell_y or cell_x differ)
        data = np.abs(df.index.to_frame()["cell_y"].diff()) + np.abs(
            df.index.to_frame()["cell_x"].diff()
        )
        # the first entry from some source
        data = data.fillna(1)
        data[data > 0] = -1
        bs.loc[df.index, ["cell_change"]] = data.astype(int)
        # use bs not data for change_log to use correct index..
        cell_change_at = data[data == -1].index
        change_iloc = []
        for _loc in cell_change_at:
            local_iloc = data.index.get_loc(_loc)
            global_iloc = bs.index.get_loc(_loc)
            if local_iloc > 0:
                global_prev_iloc = bs.index.get_loc(data.index[local_iloc - 1])
                change_iloc.append([global_iloc, global_prev_iloc])
            else:
                change_iloc.append([global_iloc, None])

        cell_c_iloc = bs.columns.get_loc("cell_change")
        count_iloc = bs.columns.get_loc("cell_change_count")
        cell_id_iloc = bs.columns.get_loc("cell_id")
        for g_iloc, g_prev_iloc in change_iloc:
            if g_prev_iloc is not None:
                # copy g_iloc,
                # replace cell_x, cell_y, cell_id with data from g_prev_iloc
                # add cell_id of g_iloc as positive value in 'cell_change'
                # do not change g_prev_iloc
                idx = bs.index[g_iloc]
                idx_prev = bs.index[g_prev_iloc]
                missing = bs.iloc[g_iloc].copy()
                missing_idx = (idx[0], idx[1], idx[2], idx_prev[3], idx_prev[4], idx[5])
                missing["event"] = "move_out"
                missing["cell_id"] = bs.iloc[g_prev_iloc, cell_id_iloc]
                missing["cell_change"] = bs.iloc[g_iloc, cell_id_iloc]
                missing["cell_change_count"] = -1
                bs_missing.append(missing.to_list())
                bs_missing_idx.append(missing_idx)

                # negative cell_id: 'node comes from this cell i.e. it was there in the previous step' -> thus increment count now!
                #
                bs.iloc[g_iloc, cell_c_iloc] = -1 * bs.iloc[g_prev_iloc, cell_id_iloc]
                bs.iloc[g_iloc, count_iloc] = 1
            else:
                # first occurrence no idea where that nodes was before set -1
                bs.iloc[g_iloc, cell_c_iloc] = -1
                bs.iloc[g_iloc, count_iloc] = 1
    missing_df = pd.DataFrame(
        bs_missing,
        index=pd.MultiIndex.from_tuples(bs_missing_idx, names=bs.index.names),
        columns=bs.columns,
    )
    bs = pd.concat([bs, missing_df])
    # remove surrounding cells (copy!)
    bs = bs.loc[pd.IndexSlice[:, :, :, cell[0], cell[1]], :].copy()

    bs = (
        bs.reset_index()
        .set_index(["table_owner", "event_time", "source_node", "pkt_count"])
        .sort_index()
    )
    for g, df in bs.groupby(by=["table_owner", "cell_x", "cell_y"]):
        bs.loc[df.index, ["cell_change_cumsum"]] = df["cell_change_count"].cumsum()
    # _m = (bs["cell_x"] == int(cell[0])) & (bs["cell_y"] == int(cell[1]))
    bs = bs.reset_index(["source_node", "pkt_count"])
    return bs


def _get_beacon_entry_exit_v2(
    b: pd.DataFrame, sim: Simulation, node_id: int, cell: tuple
):
    c_size = sim.builder.count_p.get_attribute("cell_size")
    beacon_mask = (
        (b["table_owner"] == node_id)
        & (b["cell_x"] == cell[0])
        & (b["cell_y"] == cell[1])
    )
    bs = b[beacon_mask]
    bs.loc[bs["event"] == "pre_change", ["event_id"]] = int(1)
    bs.loc[bs["event"] == "post_change", ["event_id"]] = int(2)
    bs.loc[bs["event"] == "ttl_reached", ["event_id"]] = int(3)
    bs = bs.set_index(
        ["table_owner", "event_time", "source_node", "event_id", "event_number"]
    ).sort_index()
    bs["cell_change_cumsum"] = bs["beacon_value"].cumsum()

    return (
        bs.groupby("event_time")["beacon_value"]
        .sum()
        .cumsum()
        .reset_index()
        .rename(columns=dict(beacon_value="cell_change_cumsum"))
    )


def local_measure(sim: Simulation, node_id: int, cell: tuple | None):
    if cell is None:
        return sim.builder.map_p[pd.IndexSlice[:, :, :, node_id, node_id], :]
    else:
        return sim.builder.map_p[
            pd.IndexSlice[:, int(cell[0]), int(cell[1]), node_id, node_id], :
        ]


def _merge_with_map_local(bs: pd.DataFrame, sim: Simulation, node_id: int, cell: tuple):

    # index might be not unique due to ttl check before the 'own' beacon is processed
    bs = bs.set_index(["event_number"]).sort_index()

    cumulated = bs.groupby("event_number")["beacon_value"].sum().cumsum()
    cumulated.name = "cumulated_count"
    bs = pd.merge(bs, cumulated, on="event_number")
    bs = bs.reset_index()
    bs["type"] = "neighborhoodTable"

    local = local_measure(sim, node_id, cell)
    local = local["count"].reset_index()
    local["type"] = "densityMap"
    local = local.rename(columns={"simtime": "event_time", "count": "cumulated_count"})
    bs = pd.concat([bs, local], axis=0, ignore_index=True)
    bs = bs.sort_values("event_time")
    return bs


def _get_beacon_entry_exit_v4(
    b: pd.DataFrame, sim: Simulation, node_id: int, cell: tuple
):
    beacon_mask = (
        (b["table_owner"] == node_id)
        & (b["cell_x"] == cell[0])
        & (b["cell_y"] == cell[1])
    )
    bs: pd.DataFrame = b[beacon_mask].copy(deep=True)
    bs.loc[bs["event"] == "enter_cell", ["event_id"]] = int(1)  # value +1
    bs.loc[bs["event"] == "stay_in_cell", ["event_id"]] = int(2)  # value +0
    bs.loc[bs["event"] == "leave_cell", ["event_id"]] = int(3)  # value -1
    bs.loc[bs["event"] == "ttl_reached", ["event_id"]] = int(4)  # value -1
    bs.loc[bs["event"] == "dropped", ["event_id"]] = int(5)  # value +0 (no change)
    bs = _merge_with_map_local(bs, sim, node_id, cell)
    return bs, ["source_node", "event_id", "received_at_time", "sent_time"]


def _get_beacon_entry_exit_v3(
    b: pd.DataFrame, sim: Simulation, node_id: int, cell: tuple
):
    beacon_mask = (
        (b["table_owner"] == node_id)
        & (b["cell_x"] == cell[0])
        & (b["cell_y"] == cell[1])
    )
    bs = b[beacon_mask].copy(deep=True)
    bs.loc[bs["event"] == "pre_change", ["event_id"]] = int(1)
    bs.loc[bs["event"] == "post_change", ["event_id"]] = int(2)
    bs.loc[bs["event"] == "ttl_reached", ["event_id"]] = int(3)
    bs.loc[bs["event"] == "dropped", ["event_id"]] = int(4)

    bs = _merge_with_map_local(bs, sim, node_id, cell)
    return bs, ["source_node", "event_id", "received_at_time", "sent_time"]


@threaded_lru(maxsize=64)
@timing
def get_beacon_entry_exit(sim: Simulation, node_id: int, cell: tuple):

    b = get_beacon_df(sim)
    version = 0.0
    if hasattr(b, "user_version"):
        version = b.user_version

    print(f"use beacon export version {version}")
    if version < 2.0:
        return _get_beacon_entry_exit_v0(b, sim, node_id, cell)
    elif version == 2.0:
        return _get_beacon_entry_exit_v2(b, sim, node_id, cell)
    elif version == 3.0:
        return _get_beacon_entry_exit_v3(b, sim, node_id, cell)
    else:
        return _get_beacon_entry_exit_v4(b, sim, node_id, cell)


@threaded_lru(maxsize=64)
@timing
def get_measurement_count_df(sim: Simulation, node_id, cell_id):
    x, y = get_cells(sim)[cell_id]
    df = sim.builder.map_p[pd.IndexSlice[:, float(x), float(y), :, node_id], ["count"]]
    return df.groupby(by=["simtime", "x", "y"]).count().reset_index()


@threaded_lru(maxsize=64)
@timing
def get_measurements(sim: Simulation, time, node_id, cell_id):
    # alpha = 0.5
    x, y = get_cells(sim)[cell_id]
    df = sim.builder.map_p[
        pd.IndexSlice[float(time), float(x), float(y), :, node_id], :
    ]
    return df
