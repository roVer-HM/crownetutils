import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roveranalyzer.oppanalyzer.dcd.util import DcdMetaData
from roveranalyzer.uitls import check_ax
from roveranalyzer.vadereanalyzer.plots.plots import t_cmap


class DcdMap:

    cell_features = ["count", "measured_t", "received_t", "delay", "m_aoi", "r_aoi"]

    def __init__(
        self, m_glb: DcdMetaData, id_map, glb_loc_df: pd.DataFrame, plotter=None
    ):
        self.meta = m_glb
        self.node_to_id = id_map
        self.glb_loc_df = glb_loc_df
        self.id_to_node = {v: k for k, v in id_map.items()}
        self.scenario_plotter = plotter

    def set_scenario_plotter(self, plotter):
        self.scenario_plotter = plotter

    def foo(self):
        print("bar")


class DcdMap2D(DcdMap):
    """
    decentralized crowed map
    """

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"
    tsc_global_id = 0

    @classmethod
    def from_separated_frames(cls, global_data, node_data):
        # pandas.DataFrame with [simtime, node_id] -> x,y
        # global position map for all node_ids
        glb_m, glb_df = global_data
        glb_loc_df = glb_df["node_id"].copy().reset_index()
        glb_loc_df = glb_loc_df.assign(
            node_id=glb_loc_df["node_id"].str.split(",")
        ).explode("node_id")
        # remove node_id column from global
        glb_df = glb_df.drop(labels=["node_id"], axis="columns")

        # todo compare meta

        # ID mapping
        ids = list(enumerate([n[0].node_id for n in node_data], 1))
        ids.insert(0, (0, "global"))
        node_to_id = {n[1]: n[0] for n in ids}
        id_map = pd.DataFrame(ids, columns=["ID", "node_id"]).set_index(["ID"])

        glb_loc_df["node_id"] = glb_loc_df["node_id"].apply(lambda x: node_to_id[x])
        glb_loc_df = glb_loc_df.set_index(["simtime", "node_id"])
        # merge node frames
        input_df = [(glb_m, glb_df), *node_data]
        node_dfs = []
        for meta, _df in input_df:
            if meta.node_id not in node_to_id:
                raise ValueError(f"{meta.node_id} not found in id_map")
            # add ID as column
            _df["ID"] = node_to_id[meta.node_id]
            # replace source string id with int based ID
            # todo use map (lambda not needed)
            _df["source"] = _df["source"].apply(lambda x: node_to_id[x])
            # remove index an collect in list for later pd.concat
            _df = _df.reset_index()
            node_dfs.append(_df)

        _df_ret = pd.concat(node_dfs, levels=["ID", "simtime", "x", "y"], axis=0)
        _df_ret = _df_ret.set_index(["ID", "simtime", "x", "y"])
        _df_ret = _df_ret.sort_index()

        # calculate features (delay, AoI_NR (measured-to-now), AoI_MNr (received-to-now)
        now = _df_ret.index.get_level_values("simtime")
        _df_ret["delay"] = _df_ret["received_t"] - _df_ret["measured_t"]
        # AoI_NM == measurement_age (time since last measurement)
        _df_ret["measurement_age"] = now - _df_ret["measured_t"]
        # AoI_NR == update_age (time since last update)
        _df_ret["update_age"] = now - _df_ret["received_t"]

        return cls(glb_m, node_to_id, _df_ret, glb_loc_df)

    def __init__(
        self,
        m_glb: DcdMetaData,
        id_map,
        tc2d_df: pd.DataFrame,
        glb_loc_df: pd.DataFrame,
        plotter=None,
    ):
        super().__init__(m_glb, id_map, glb_loc_df, plotter)
        self.raw2d = tc2d_df

    def iter_nodes_d2d(self, first=0, last=-1):
        _i = pd.IndexSlice

        data = self.raw2d.loc[_i[first:], :].groupby(level=self.tsc_id_idx_name)
        for i, ts_2d in data:
            yield i, ts_2d

    @property
    def glb_2d(self):
        return self.raw2d.loc[
            (self.tsc_global_id),
        ]

    def age_mask(self, age_column, threshold):
        _mask = self.raw2d[age_column] <= threshold
        return _mask

    def with_age(self, age_column, threshold):
        df2d = self.raw2d[self.age_mask(age_column, threshold)].copy()
        return DcdMap2D(self.meta, self.node_to_id, df2d, self.scenario_plotter)

    def count_diff(self, diff_metric="diff"):
        metric = {
            "diff": lambda loc, glb: glb - loc,
            "abs_diff": lambda loc, glb: (glb - loc).abs(),
            "sqr_diff": lambda loc, glb: (glb - loc) ** 2,
        }
        _df = pd.DataFrame(self.raw2d.groupby(["ID", "simtime"]).sum()["count"])
        _df = _df.combine(_df.loc[0], metric[diff_metric])
        _df = _df.rename(columns={"count": "count_diff"})
        return _df

    def plot_count(self, ax=None):
        f, ax = check_ax(ax)
        ax.set_title("Total node count over time")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("total node count")

        for id, df in self.iter_nodes_d2d(first=1):
            df_time = df.groupby(level=self.tsc_time_idx_name).sum()
            ax.plot(
                df_time.index, df_time["count"], label=f"{id}_{self.id_to_node[id]}"
            )

        g = self.glb_2d.groupby(level=self.tsc_time_idx_name).sum()
        ax.plot(
            g.index.get_level_values(self.tsc_time_idx_name),
            g["count"],
            label=f"0_{self.id_to_node[0]}",
        )
        ax.legend()
        return ax

    def plot_count_diff(self, ax=None):
        f, ax = check_ax(ax)
        ax.set_title("Node count over Error time")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("MES of count")
        _i = pd.IndexSlice
        nodes = (
            self.raw2d.loc[_i[1:], _i["count"]]
            .groupby(level=[self.tsc_id_idx_name, self.tsc_time_idx_name])
            .sum()
            .groupby(level="simtime")
            .mean()
        )
        glb = self.glb_2d.groupby(level=self.tsc_time_idx_name).sum()["count"]
        ax.plot(nodes.index, nodes, label="mean node count")
        ax.plot(glb.index, glb, label="actual node count")
        f.legend()
        return ax

    def plot_map(self, time_step, id, ax=None):
        _i = pd.IndexSlice
        data = self.raw2d.loc[_i[id, time_step], ["count"]]
        full_index = self.meta.grid_index_2d(real_coords=True)
        df = pd.DataFrame(np.zeros((len(full_index), 1)), columns=["count"])
        df = df.set_index(full_index)
        df.update(data)
        df = df.unstack().T
        f, ax = check_ax(ax)

        ax.set_title(
            f"Density map for node {id}_{self.id_to_node[id]} at time {time_step}"
        )
        ax.set_aspect("equal")

        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)
        cmap = t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0))
        pcm = ax.pcolormesh(self.meta.X, self.meta.Y, df, cmap=cmap, shading="auto")
        f.colorbar(pcm, ax=ax, shrink=0.5)

        return ax

    def plot_location_map(self, time_step, ax=None):
        _i = pd.IndexSlice
        own_cell_mask = self.raw2d["own_cell"] == 1
        places = (
            self.raw2d[
                own_cell_mask
            ]  # only own cells (cells where one node is located)
            .index.to_frame()  # make the index to the dataframe
            .reset_index(["x", "y"], drop=True)  # remove not needed index
            .drop(
                columns=["ID", "simtime"]
            )  # and remove columns created by to_frame we do not need
        )
        places = places.loc[_i[:, time_step], :]  # select only the needed timestep

        f, ax = check_ax(ax)

        ax.set_title(f"Node Placement at time {time_step}s")
        ax.set_aspect("equal")
        ax.set_xlim([0, self.meta.x_dim])
        ax.set_ylim([0, self.meta.y_dim])
        if self.scenario_plotter is not None:
            self.scenario_plotter.add_obstacles(ax)

        for id, df in places.groupby(level=self.tsc_id_idx_name):
            # move coordinate for node :id: to center of cell.
            df = df + 0.5 * self.meta.cell_size
            ax.scatter(df["x"], df["y"], label=f"{id}_{self.id_to_node[id]}")

        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        return ax

    def plot2(self, simtime, id, title=""):
        f, ax = plt.subplots(2, 2, figsize=(16, 9))
        ax = ax.flatten()
        f.suptitle(f"Node {id}_{self.id_to_node[id]} for time {simtime} {title}")
        self.plot_map(simtime, id, ax[0])
        self.plot_location_map(simtime, ax[1])
        self.plot_count(ax[2])
        ax[3].clear()
        return f

    def describe_raw(self, global_only=False):
        _i = pd.IndexSlice
        if global_only:
            data = self.raw2d.loc[_i[0], :]  # only global data
            data_str = "Global"
        else:
            data = self.raw2d.loc[_i[1:], :]  # all *but* global
            data_str = "Local"

        desc = data.describe().T
        print("=" * 79)
        print(f"Counts with values > 0 ({data_str}):")
        print(desc.loc[["count"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Delay for each cell ({data_str}):")
        print(desc.loc[["delay"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Time since measurement was taken ({data_str}):")
        print(desc.loc[["measurement_age"], ["mean", "std", "min", "max"]])
        print("-" * 79)
        print(f"Time since last update ({data_str}):")
        print(desc.loc[["update_age"], ["mean", "std", "min", "max"]])
        print("=" * 79)

    def age_error(self, age_column):
        """
        What is the error in the number of counts for each map
        """

        return 1
