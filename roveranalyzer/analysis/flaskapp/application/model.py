from __future__ import annotations

import collections
import json
import shutil
from functools import lru_cache
from os.path import basename, join, split

import numpy as np
import pandas as pd
from numpy import maximum
from pandas import DataFrame, MultiIndex

from roveranalyzer.analysis.dashapp import DashUtil
from roveranalyzer.analysis.flaskapp.application import cache
from roveranalyzer.analysis.flaskapp.application.utils import threaded_lru
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.simulators.opp.provider.hdf.DcDGlobalPosition import (
    DcdGlobalDensity,
    DcdGlobalPosition,
    pos_density_from_csv,
)
from roveranalyzer.simulators.opp.provider.hdf.DcdMapCountProvider import DcdMapCount
from roveranalyzer.simulators.opp.provider.hdf.DcdMapProvider import DcdMapProvider
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import (
    BaseHdfProvider,
    FrameConsumer,
)
from roveranalyzer.simulators.opp.scave import CrownetSql
from roveranalyzer.simulators.vadere.plots.scenario import Scenario
from roveranalyzer.utils.general import Project
from roveranalyzer.utils.logging import timing


class Simulation:
    def __init__(self, data_root, label):
        self.label = label
        self.data_root, self.builder, self.sql = OppAnalysis.builder_from_output_folder(
            data_root
        )
        self.pos: BaseHdfProvider = BaseHdfProvider(
            join(self.data_root, "trajectories.h5"), group="trajectories"
        )

    @staticmethod
    def asset_pdf_path(name, base="assets/pdf", suffix=""):
        base_name = basename(name).replace(".pdf", "")
        return join(base, f"{base_name}{suffix}.pdf")

    @classmethod
    def copy_pdf(cls, name, sim: Simulation, base):
        shutil.copyfile(
            src=join(sim.data_root, name),
            dst=cls.asset_pdf_path(name, base=base, suffix=sim.label),
        )


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
    name = split(sim.sql.vadere_scenario)[-1]
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
def get_node_ids_for_cell(sim: Simulation, cell_id):
    ca = get_cell_error_data(sim, cell_id)
    return ca["ID"].unique()


@threaded_lru(maxsize=16)
@timing
def get_beacon_df(sim: Simulation):
    b = pd.read_csv(join(sim.data_root, "beacons.csv"), delimiter=";")
    if any(b["cell_x"] > 0x0FFF_FFFF) or any(b["cell_y"] > 0x0FFF_FFFF):
        raise ValueError("cell positions exceed 28 bit.")
    bound = sim.builder.count_p.get_attribute("cell_bound")
    b["posY"] = bound[1] - b["posY"]  # fix translation !
    b["cell_id"] = (b["cell_x"].values << 28) + b["cell_y"]
    return b


@threaded_lru(maxsize=64)
@timing
def get_beacon_entry_exit(sim: Simulation, node_id: int, cell: tuple):

    b = get_beacon_df(sim)

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
        ["table_owner", "source_node", "event_time", "cell_x", "cell_y"]
    ).sort_index()
    bs["cell_change"] = 0
    bs["cell_change_count"] = 0
    bs["cell_change_cumsum"] = 0
    if bs.empty:
        return bs
    bs_missing = []
    bs_missing_idx = []
    for g, df in bs.groupby(by=["table_owner", "source_node"]):
        data = np.abs(df.index.to_frame()["cell_y"].diff()) + np.abs(
            df.index.to_frame()["cell_x"].diff()
        )
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
                missing_idx = (idx[0], idx[1], idx[2], idx_prev[3], idx_prev[4])
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
        .set_index(["table_owner", "event_time", "source_node"])
        .sort_index()
    )
    for g, df in bs.groupby(by=["table_owner", "cell_x", "cell_y"]):
        bs.loc[df.index, ["cell_change_cumsum"]] = df["cell_change_count"].cumsum()
    # _m = (bs["cell_x"] == int(cell[0])) & (bs["cell_y"] == int(cell[1]))
    bs = bs.reset_index("source_node")
    return bs


@threaded_lru(maxsize=64)
@timing
def get_measurement_count_df(sim: Simulation, node_id, cell_id):
    x, y = get_cells(sim)[cell_id]
    df = sim.builder.map_p[pd.IndexSlice[:, float(x), float(y), :, node_id], ["count"]]
    return df.groupby(by=["simtime", "x", "y"]).count().reset_index()


@threaded_lru(maxsize=64)
@timing
def get_measurements(sim: Simulation, time, node_id, cell_id):
    alpha = 0.5
    # todo
    # alpha = sim.get_config()[".....mapCfg"]["alpha"]
    x, y = get_cells(sim)[cell_id]
    df = sim.builder.map_p[
        pd.IndexSlice[float(time), float(x), float(y), :, node_id], :
    ]
    dist_sum = df["sourceEntry"].sum()
    time_sum = df["measurement_age"].sum()
    # df["ymfD_t_old"] = alpha*(df["measurement_age"]/time_sum)

    t_min = df["measurement_age"].min()
    time_sum_new = df["measurement_age"].sum() - df["measurement_age"].shape[0] * t_min

    df["ymfD_t"] = alpha * ((df["measurement_age"] - t_min) / time_sum_new)
    df["ymfD_d"] = (1 - alpha) * (df["sourceEntry"] / dist_sum)
    df["ymfD"] = df["ymfD_t"] + df["ymfD_d"]

    return df
