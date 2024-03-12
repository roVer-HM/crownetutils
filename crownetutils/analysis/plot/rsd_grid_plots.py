from __future__ import annotations

import os
import shutil
from abc import ABC
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, ScalarFormatter

from crownetutils.analysis.common import RunMap, Simulation
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgBuilder, DpmmCfgDb
from crownetutils.analysis.dpmm.dpmm_sql import DpmmSql
from crownetutils.analysis.hdf_providers.map_age_over_distance import (
    MapMeasurementsAgeOverDistance,
    MapSizeAndAgeOverDistance,
)
from crownetutils.analysis.hdf_providers.map_error_data import MapCountError
from crownetutils.analysis.hdf_providers.node_position import NodePositionWithRsdHdf
from crownetutils.analysis.hdf_providers.node_rx_data import NodeRxData
from crownetutils.analysis.hdf_providers.node_tx_data import NodeTxData
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.utils.logging import logger
from crownetutils.utils.plot import (
    GridPlot,
    GridPlotIter,
    PlotUtil,
    PlotUtil_,
    calc_box_stats,
    percentile,
)


class RsdGridPlotterStyler(ABC):
    def get_output_path(self, file_name: str) -> str:
        ...

    def get_suptitle(self, suptitle: str) -> str:
        ...

    def is_save_and_close(self) -> bool:
        ...


class EmptyStyler(RsdGridPlotterStyler):
    def get_output_path(self, file_name: str) -> str:
        raise NotImplementedError()

    def get_suptitle(self, suptitle: str) -> str:
        raise NotImplementedError()

    def is_save_and_close(self) -> bool:
        return False


class RunMapRsdGridPlotStyler(RsdGridPlotterStyler):
    @classmethod
    def plot_kwargs(cls, run_map, parameter, seed):
        obj = cls(run_map, parameter, seed)
        return {"sim": run_map[parameter][seed], "plot_styler": obj}

    def __init__(self, run_map: RunMap, parameter: str, seed: int) -> None:
        super().__init__()
        self.run_map = run_map
        self.parameter = parameter
        self.seed = seed

    def get_output_path(self, file_name: str) -> str:
        p = file_name.split(".")
        if len(p) == 1:
            _name = f"{file_name}_{self.parameter}_{self.seed}"
        else:
            _name = f"{p[0]}_{self.parameter}_{self.seed}.{p[1]}"
        return self.run_map.path(_name)

    def get_suptitle(self, suptitle: str) -> str:
        return f"{suptitle} - {self.parameter}"

    def is_save_and_close(self) -> bool:
        return False


class RsdGridPlotter(PlotUtil_):
    @classmethod
    def from_sim(
        cls,
        sim: Simulation,
        rows: int = 3,
        cols: int = 5,
        single_fig_width: float = 4.0,
        rsd_filter: List[int] | None = None,
    ):
        pos = NodePositionWithRsdHdf(sim, hdf_path=sim.dpmm_cfg.position.path)
        if rsd_filter is None:
            colors = pos.enb_colors(with_zero=False)
        else:
            colors = {
                k: v
                for k, v in pos.enb_colors(with_zero=False).items()
                if k in rsd_filter
            }

        return cls(
            color_map=colors,
            rows=rows,
            cols=cols,
            single_fig_width=single_fig_width,
            rsd_filter=rsd_filter,
        )

    def __init__(
        self,
        color_map,
        rows: int = 3,
        cols: int = 5,
        single_fig_width: float = 4.0,
        rsd_filter: List[int] | None = None,
    ) -> None:
        super().__init__()
        self.color_map = color_map
        self.rsd_filter = rsd_filter
        color_list = [color_map[c] for c in self.get_rsd_list()]
        self.create_figure = self.create_figure_builder(
            color_list, rows, cols, single_fig_width
        )

    def get_node_pos(self, sim: Simulation) -> NodePositionWithRsdHdf:
        obj = NodePositionWithRsdHdf(sim, hdf_path=sim.dpmm_cfg.position.path)
        return obj

    def save_and_close_grid_figure(
        self, grid_figure_iter: GridPlotIter, suptitle, path
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        grid_figure_iter.fig.suptitle(suptitle)
        grid_figure_iter.fig.tight_layout()
        grid_figure_iter.fig.savefig(path)
        grid_figure_iter.close()
        logger.info(f"saved and closed figure to {path}")

    def get_rsd_list(self):
        if self.rsd_filter:
            return self.rsd_filter
        else:
            return list(np.arange(len(self.color_map)) + 1)

    def rsd_ax_iter(self, iter: GridPlotIter):
        rsd_list = self.get_rsd_list()
        if len(rsd_list) != len(iter):
            raise ValueError(
                f"RSD list and axes iter do not match {len(rsd_list)} != {iter}"
            )

        return zip(rsd_list, iter)

    @classmethod
    def create_figure_builder(
        cls, color_list, rows: int = 3, cols: int = 5, single_fig_width: float = 4.0
    ):
        if len(color_list) != rows * cols:
            raise ValueError(f"need {rows*cols} colors got {len(color_list)}")

        def figure_builder():
            fig_size = (single_fig_width * cols, 9 / 16 * single_fig_width * rows)
            # get list of colors based on list of rsd's
            grid_figure = GridPlot(
                rows=rows, columns=cols, colors=color_list, figsize=fig_size
            )
            grid_figure_iter: GridPlotIter = grid_figure.iter_lowerLeftOrig()
            return grid_figure_iter

        return figure_builder

    def get_box_props(self, rsd_color):
        flierprops = dict(
            marker="2",
            markerfacecolor=rsd_color,
            markeredgecolor=rsd_color,
            markersize=1,
            linestyle="none",
        )
        medianprops = dict(linestyle="-", linewidth=2.0, color="firebrick")
        meanprops = dict(linestyle="dotted", linewidth=2, color="green")
        return dict(flierprops=flierprops, medianprops=medianprops, meanprops=meanprops)

    def box_legend(self, rsd_color):
        props = self.get_box_props(rsd_color)
        median_line = Line2D(
            [0], [0], label="median", color=props["medianprops"]["color"]
        )
        mean_line = Line2D(
            [0],
            [0],
            linestyle="dotted",
            label="mean",
            color=props["meanprops"]["color"],
        )
        fliers = Line2D(
            [0],
            [0],
            label="fliers",
            marker="2",
            markersize=1,
            markeredgecolor=rsd_color,
            markerfacecolor=rsd_color,
            linestyle="",
        )
        return [median_line, mean_line, fliers]

    def plot_rsd_size_grid(self, sim: Simulation, plot_styler: RsdGridPlotterStyler):
        map_err = MapCountError(hdf_path=sim.dpmm_cfg.map_count_error.path)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            data = map_err.hdf_map_measure_rsd_local.select(
                where=f"rsd_id = {rsd}"
            ).reset_index()
            ax.plot(
                "simtime",
                "map_glb_count",
                data=data,
                color="black",
                label="total_nodes",
            )
            self.fill_between(
                data=data,
                x="simtime",
                val="map_mean_count",
                fill_val=["map_count_p25", "map_count_p75"],
                plot_lbl="mean node count",
                line_args=dict(color=color),
                fill_args=dict(label="Q1/Q3"),
                ax=ax,
            )
            ax.set_title(f"Rsd {rsd}")
            ax.legend()
            ax.set_ylim((0, 200))
            self.auto_major_minor_locator(ax)

        for ax in grid_figure_iter.left_axes():
            ax.set_ylabel("Num nodes")

        for ax in grid_figure_iter.lower_axes():
            ax.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path(f"rsd_size_grid.png")
            suptitle = plot_styler.get_suptitle(
                f"Number of nodes in RSD total and seen by average density map (bin_size = {1.0}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_serving_enb_grid(
        self, sim: Simulation, plot_styler: RsdGridPlotterStyler, bin_size=1.0
    ):
        sql = sim.sql

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            ax: plt.Axes
            data = OppAnalysis.get_avgServedBlocksUl(
                sql, enb_index=rsd - 1
            )  # uses index! not ID
            t = f"Rsd {rsd}:  {self.title_stats(data['value'])}RBs"

            data = self.append_bin(
                data=data,
                bin_size=bin_size,
                agg=["mean", percentile(25), percentile(75)],
            )
            self.fill_between(
                data=data,
                x="bin_right",
                val="value_mean",
                fill_val=["value_p25", "value_p75"],
                ax=ax,
                line_args=dict(color=color),
                plot_lbl="mean RBs",
                fill_args=dict(label="Q1/3"),
            )
            ax.set_title(t)
            ax.set_ylim(0, 27)
            ax.yaxis.set_major_locator(MultipleLocator(5))
            ax.yaxis.set_minor_locator(MultipleLocator(5 / 4))
            ax.legend()
            self.auto_major_minor_locator(ax, what="x")

        grid_figure_iter.set_ylabel("Num RBs")
        grid_figure_iter.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path(f"enb_grid.png")
            suptitle = plot_styler.get_suptitle(
                f"Number Resource blocks used (bin size = {bin_size}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_throughput_grid(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        app: str = "e_map",
        bin_size: float = 10.0,
    ):
        ntx: NodeTxData = NodeTxData(sim.dpmm_cfg.node_tx.path)

        tp_target_rate = ntx.get_target_rates(1 / 8)  # B/s

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            ax: plt.Axes

            tp = ntx.get_tx_throughput_diff_by_app(
                target_rates=tp_target_rate,
                bin_size=bin_size,
                throughput_unit=1000,
                serving_enb=rsd,
            )
            ax.plot("time", app, data=tp.reset_index(), label=f"rsd {rsd}", color=color)
            ax.hlines(
                tp_target_rate[app] / 1000,
                xmin=0,
                xmax=1000,
                color="red",
                label="target rate",
            )

            ax.set_title(f"Rsd {rsd}")
            ax.set_ylim(0, 140)
            self.auto_major_minor_locator(ax)

        grid_figure_iter.set_ylabel("throughput in kB/s")
        grid_figure_iter.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("e_map_throughput_grid.png")
            suptitle = plot_styler.get_suptitle(
                f"Throughput in KB/s for  entropy maps (bin size = {bin_size}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_packet_delay_grid(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        app: str = "e_map",
        bin_size: float = 10.0,
    ):
        rx: NodeRxData = NodeRxData(hdf_path=sim.dpmm_cfg.node_rx.path)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            delay = (
                rx.rcvd_data(app)
                .select(where=f"servingEnb=={rsd}", columns=["delay"])
                .reset_index()[["time", "delay"]]
                .copy()
            )
            t_stat = self.title_stats(delay["delay"], ",.2e", ",.2e")
            t = f"Rsd {rsd}:  {t_stat}s"

            delay = self.append_bin(
                data=delay,
                idx_name="time",
                bin_size=bin_size,
                start=0,
                agg=["mean", percentile(25), percentile(75)],
            )
            self.fill_between(
                data=delay,
                x="bin_right",
                val="delay_mean",
                fill_val=["delay_p25", "delay_p75"],
                plot_lbl="mean delay",
                line_args=dict(color=color),
                fill_args=dict(label="Q1/3"),
                ax=ax,
            )
            ax.legend()
            ax.set_title(t)

        grid_figure_iter.set_ylabel("Delay in seconds")
        grid_figure_iter.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("e_map_delay_grid.png")
            suptitle = plot_styler.get_suptitle(
                f"Delay in seconds for entropy maps (bin size = {bin_size}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_burst_information_ratio_grid(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        app: str = "e_map",
        bin_size: float = 10.0,
    ):
        ntx = NodeTxData(sim.dpmm_cfg.node_tx.path)

        cfg = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        if cfg.map_size.exists():
            map_size = pd.read_csv(cfg.map_size.path, index_col=None)
        else:
            map_sql = DpmmSql(DpmmCfgBuilder.load_entropy_cfg(sim.data_root))
            map_size = map_sql.get_cell_count_by_host_id_over_time()

        burst = ntx.get_information_transfer_per_burst(app=app, map_size_data=map_size)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            ax: plt.Axes
            _rsd_data = burst[burst["servingEnb"] == rsd].reset_index()

            _data = self.append_bin(
                data=_rsd_data[["time", "information_transfer_per_burst"]].copy(),
                idx_name="time",
                bin_size=bin_size,
                start=0.0,
                agg=["mean", "median", percentile(25), percentile(75)],
            )

            self.fill_between(
                data=_data,
                x="bin_right",
                val="information_transfer_per_burst_mean",
                fill_val=[
                    "information_transfer_per_burst_p25",
                    "information_transfer_per_burst_p75",
                ],
                ax=ax,
                plot_lbl="mean",
                fill_args=dict(label="Q1/3"),
                line_args=dict(color=color),
            )

            t = f"Rsd {rsd}: Map ratio  {self.title_stats(_data['information_transfer_per_burst_mean'])}"
            ax.set_title(t)
            ax.legend(loc="lower right")

            self.auto_major_minor_locator(ax)
            ax.set_ylim(0, 1.1)

        grid_figure_iter.set_ylabel("map ratio")
        grid_figure_iter.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("e_map_information_transfer_grid.png")
            suptitle = plot_styler.get_suptitle(
                f"Map ratio communicated in single burst bin size = {bin_size}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_map_age_over_distance_grid(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        distance_bin: float = 100.0,
    ):
        # need entropy config.
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapMeasurementsAgeOverDistance(
            hdf_path=e_cfg.map_measurements_age_over_distance.path
        )

        grid_figure_iter: GridPlotIter = self.create_figure()
        metric_map = m.metric_map
        metric_id = metric_map["age_of_information"]

        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            data = m.hdf_age_over_dist_rsd.select(
                where=f"rsd={rsd} and metric={metric_id}"
            ).reset_index()
            t_stat = self.title_stats(data["mean"], ",.2e", ",.2e")
            data["dist_bin"] = np.floor(data["dist_left"] / distance_bin) * distance_bin
            data = (
                data.groupby("dist_bin")["mean"]
                .agg(calc_box_stats(use_mpl_name=True, include_mean=True))
                .to_frame()
            )

            boxes = []
            for t, box in data.iterrows():
                b = box["mean"]
                b["label"] = t
                boxes.append(b)

            b = ax.bxp(
                boxes,
                positions=[float(_b["label"]) for _b in boxes],
                widths=distance_bin * 0.9,
                showmeans=True,
                meanline=True,
                **self.get_box_props(rsd_color=color),
            )
            # replace FixedFormatter provided by bxp(...)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.xaxis.set_minor_locator(MultipleLocator(200))
            ax.set_xlim(0, 4000)

            # ax.scatter("dist_left", "mean", data=data, color=color, marker=".")
            t = f"Rsd {rsd}: AoI  {t_stat}s"
            ax.set_title(t)

        grid_figure_iter.set_ylabel("AoI in seconds")
        grid_figure_iter.set_xlabel("Distance of cell and measurement owner in meter)")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("age_of_distance.png")
            suptitle = plot_styler.get_suptitle(
                f"Age of Information (AoI)  over distance (time bin= 10.0s / dist bin = {distance_bin}m)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def plot_map_number_measures_over_distance_grid(
        self, sim: Simulation, plot_styler: RsdGridPlotterStyler
    ):
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapMeasurementsAgeOverDistance(
            hdf_path=e_cfg.map_measurements_age_over_distance.path
        )

        metric_map = m.hdf_age_over_dist.get_attribute("metric_map")
        metric_id = metric_map["age_of_information"]

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            data = m.hdf_age_over_dist_rsd.select(
                where=f"rsd={rsd} and metric={metric_id}"
            ).reset_index()
            ax.scatter("dist_left", "count", data=data, color=color, marker=".")
            t_stat = self.title_stats(data["count"], ",.2e", ",.2e")
            t = f"Rsd {rsd}: {t_stat}s"
            ax.set_title(t)
            self.auto_major_minor_locator(ax)

        for a in grid_figure_iter.flat_iter():
            a.set_xlim(0, 4100)
            a.xaxis.set_major_locator(MultipleLocator(1000))
            a.xaxis.set_minor_locator(MultipleLocator(200))

        grid_figure_iter.set_ylabel("Number of Measures")
        grid_figure_iter.set_xlabel("Distance of cell and measurement owner in meter)")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("num_measures_over_distance.png")
            suptitle = plot_styler.get_suptitle(
                f"Number of measures over distance (time bin= 10.0s / dist bin = 50.0m) - {parameter}"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def data_map_size_over_time(self, all_data, rsd, bin_size):
        data = all_data.loc[rsd].groupby("simtime").mean().reset_index()
        data["simtime"] = np.floor(data["simtime"] / bin_size) * bin_size
        data = (
            data.groupby("simtime")["number_of_cells"]
            .agg(calc_box_stats(use_mpl_name=True, include_mean=True))
            .to_frame()
        )

        boxes = []
        for t, box in data.iterrows():
            b = box["number_of_cells"]
            b["label"] = t
            boxes.append(b)
        return data, boxes

    def plot_map_size_over_time(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        seed=0,
        bin_size: float = 20.0,
    ):
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)

        if e_cfg.map_size.exists():
            average_map_size = pd.read_csv(e_cfg.map_size.path, index_col=None)
        else:
            map_sql = DpmmSql(DpmmCfgBuilder.load_entropy_cfg(sim.data_root))
            average_map_size = map_sql.get_cell_count_by_host_id_over_time()

        pos = self.get_node_pos(sim)
        average_map_size = pos.merge_rsd_id_on_host_time_interval(
            data=average_map_size, host_id_col="host_id", time_col="simtime"
        )
        average_map_size = average_map_size.set_index(
            ["servingEnb", "simtime"]
        ).sort_index()

        map_err = MapCountError(hdf_path=sim.dpmm_cfg.map_count_error.path)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            _, boxes = self.data_map_size_over_time(average_map_size, rsd, bin_size)
            ax: plt.Axes
            b = ax.bxp(
                boxes,
                positions=[float(_b["label"]) for _b in boxes],
                widths=bin_size * 0.9,
                showmeans=True,
                meanline=True,
                **self.get_box_props(rsd_color=color),
            )
            ax.xaxis.set_major_formatter(ScalarFormatter())

            ax2 = ax.twinx()
            data_size = map_err.hdf_map_measure_rsd_local.select(
                where=f"rsd_id = {rsd}"
            ).reset_index()
            ax2.plot(
                "simtime",
                "map_glb_count",
                data=data_size,
                color="black",
                label="total_nodes",
            )
            self.fill_between(
                data=data_size,
                x="simtime",
                val="map_mean_count",
                fill_val=["map_count_p25", "map_count_p75"],
                plot_lbl="mean node count",
                line_args=dict(color=color),
                fill_args=dict(label="Q1/Q3"),
                ax=ax2,
            )

            self.auto_major_minor_locator(ax)

            t = f"Rsd {rsd}"
            ax.set_title(t)
            ax.legend(handles=self.box_legend(color), loc="upper right")

        grid_figure_iter.set_ylabel("Map Size")
        grid_figure_iter.set_xlabel("Simulation time in seconds")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("num_cells_over_distance.png")
            suptitle = plot_styler.get_suptitle(
                f"Average map size over time  (time bin= {bin_size}s)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter

    def data_map_size_over_distance(
        self,
        provider: MapSizeAndAgeOverDistance,
        distance_bin: float,
        rsd: int,
    ) -> List[dict]:
        data = provider.hdf_rsd.select(
            where=f"rsd={rsd} and metric=0", columns=["count"]
        )
        # use 100m bins not 50
        data["dist_bin"] = (
            np.floor(data.index.get_level_values("dist_left") / distance_bin)
            * distance_bin
        )

        stats = (
            data.groupby(["dist_bin"])["count"]
            .aggregate(calc_box_stats(include_mean=True, use_mpl_name=True))
            .to_frame()
        )
        boxes = []
        for idx, df in stats.iterrows():
            b = df["count"]
            b["label"] = idx
            boxes.append(b)
        return boxes

    def plot_map_size_over_distance_grid(
        self,
        sim: Simulation,
        plot_styler: RsdGridPlotterStyler,
        distance_bin: float = 100.0,
    ):
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapSizeAndAgeOverDistance(
            hdf_path=e_cfg.map_size_and_age_over_distance.path
        )

        grid_figure_iter: GridPlotIter = self.create_figure()

        for rsd, (_, ax, color) in self.rsd_ax_iter(grid_figure_iter):
            boxes = self.data_map_size_over_distance(m, distance_bin, rsd)

            ax.bxp(
                boxes,
                positions=[float(_b["label"]) for _b in boxes],
                widths=100 * 0.85,
                showmeans=True,
                meanline=True,
                **self.get_box_props(rsd_color=color),
            )
            # replace FixedFormatter provided by bxp(...)
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_major_locator(MultipleLocator(1000))
            ax.xaxis.set_minor_locator(MultipleLocator(200))
            ax.set_xlim(0, 4000)

            t = f"Rsd {rsd}"
            ax.set_title(t)
            ax.legend(handles=self.box_legend(color), loc="upper right")

        grid_figure_iter.set_ylabel("Number of cells in map")
        grid_figure_iter.set_xlabel("Distance of cell and measurement owner in meter")

        if plot_styler.is_save_and_close():
            path = plot_styler.get_output_path("map_size_over_distance.png")
            suptitle = plot_styler.get_suptitle(
                f"Number of cells in distance interval (time bin= 1.0s / dist bin = 100.0m)"
            )
            self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
        else:
            return grid_figure_iter.fig, grid_figure_iter
