import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator

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
    PlotUtil_,
    box_stats_to_plt_box,
    calc_box_stats,
    percentile,
    percentiles_dict,
)


class RsdGridPlotter(PlotUtil_):
    @classmethod
    def from_sim(
        cls,
        sim: Simulation,
        rows: int = 3,
        cols: int = 5,
        single_fig_width: float = 4.0,
    ):
        pos = NodePositionWithRsdHdf(sim, hdf_path=sim.dpmm_cfg.position.path)
        return cls(
            color_map=pos.enb_colors(with_zero=False),
            rows=rows,
            cols=cols,
            single_fig_width=single_fig_width,
        )

    def __init__(
        self, color_map, rows: int = 3, cols: int = 5, single_fig_width: float = 4.0
    ) -> None:
        super().__init__()
        self.color_map = color_map
        self.create_figure = self.create_figure_builder(
            color_map, rows, cols, single_fig_width
        )
        self.rsd_filter = None

    def get_node_pos(sim: Simulation) -> NodePositionWithRsdHdf:
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
            return list(np.arrange(len(self.color_map)) + 1)

    @classmethod
    def create_figure_builder(
        self, colors, rows: int = 3, cols: int = 5, single_fig_width: float = 4.0
    ):
        if len(colors) != rows * cols:
            raise ValueError(f"need {rows*cols} colors got {len(colors)}")

        def figure_builder():
            fig_size = (single_fig_width * 5, 9 / 16 * single_fig_width * 3)
            grid_figure = GridPlot.grid_3x5(colors=colors, figsize=fig_size)
            grid_figure_iter: GridPlotIter = grid_figure.iter_lowerLeftOrig()
            return grid_figure_iter

        return figure_builder

    def plot_rsd_size_grid(
        self, run_map: RunMap, parameter="insertionOrder_ttl15", seed=0
    ):
        sim: Simulation = run_map[parameter][seed]

        map_err = MapCountError(hdf_path=sim.dpmm_cfg.map_count_error.path)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
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

        path = run_map.path(
            parameter.split("_")[0], f"rsd_size_grid_{parameter}_{seed}.png"
        )
        suptitle = f"Number of nodes in RSD total and seen by average density map (bin_size = {1.0}s) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_serving_enb_grid(
        self, run_map: RunMap, bin_size=1.0, parameter="insertionOrder_ttl15", seed=0
    ):
        sim: Simulation = run_map[parameter][seed]
        sql = sim.sql

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            ax: plt.Axes
            rsd = idx + 1
            data = OppAnalysis.get_avgServedBlocksUl(
                sql, enb_index=idx
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

        path = run_map.path(parameter.split("_")[0], f"enb_grid_{parameter}_{seed}.png")
        suptitle = f"Number Resource blocks used (bin size = {bin_size}s) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_throughput_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]

        ntx: NodeTxData = NodeTxData(sim.dpmm_cfg.node_tx.path)

        tp_target_rate = ntx.get_target_rates(1 / 8)  # B/s

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
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

        path = run_map.path(
            parameter.split("_")[0], f"e_map_throughput_grid_{parameter}_{seed}.png"
        )
        suptitle = f"Throughput in KB/s for  entropy maps (bin size = {bin_size}s) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_packet_delay_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]

        rx: NodeRxData = NodeRxData(hdf_path=sim.dpmm_cfg.node_rx.path)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
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

        path = run_map.path(
            parameter.split("_")[0], f"e_map_delay_grid_{parameter}_{seed}.png"
        )
        suptitle = (
            f"Delay in seconds for entropy maps (bin size = {bin_size}s) - {parameter}"
        )
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_burst_information_ratio_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]

        ntx = NodeTxData(sim.dpmm_cfg.node_tx.path)

        map_sql = DpmmSql(DpmmCfgBuilder.load_entropy_cfg(sim.data_root))
        map_size = map_sql.get_cell_count_by_host_id_over_time()

        burst = ntx.get_information_transfer_per_burst(app=app, map_size_data=map_size)

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
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

        path = run_map.path(
            parameter.split("_")[0],
            f"e_map_information_transfer_grid_{parameter}_{seed}.png",
        )
        suptitle = f"Map ratio communicated in single burst bin size = {bin_size}s)"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_map_age_over_distance_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]
        # need entropy config.
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapMeasurementsAgeOverDistance(
            hdf_path=e_cfg.map_measurements_age_over_distance.path
        )

        grid_figure_iter: GridPlotIter = self.create_figure()
        metric_map = m.metric_map
        metric_id = metric_map["age_of_information"]

        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
            data = m.hdf_age_over_dist_rsd.select(
                where=f"rsd={rsd} and metric={metric_id}"
            ).reset_index()
            ax.scatter("dist_left", "mean", data=data, color=color, marker=".")
            t_stat = self.title_stats(data["mean"], ",.2e", ",.2e")
            t = f"Rsd {rsd}: AoI  {t_stat}s"
            ax.set_title(t)
            self.auto_major_minor_locator(ax)

        for a in grid_figure_iter.flat_iter():
            a.set_xlim(0, 4100)
            a.xaxis.set_major_locator(MultipleLocator(1000))
            a.xaxis.set_minor_locator(MultipleLocator(200))
        grid_figure_iter.set_ylabel("AoI in seconds")
        grid_figure_iter.set_xlabel("Distance of cell and measurement owner in meter)")

        path = run_map.path(
            parameter.split("_")[0], f"age_of_distance_{parameter}_{seed}.png"
        )
        suptitle = f"Age of Information (AoI)  over distance (time bin= 10.0s / dist bin = 50.0m) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_map_number_measures_over_distance_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapMeasurementsAgeOverDistance(
            hdf_path=e_cfg.map_measurements_age_over_distance.path
        )

        metric_map = m.hdf_age_over_dist.get_attribute("metric_map")
        metric_id = metric_map["age_of_information"]

        grid_figure_iter: GridPlotIter = self.create_figure()
        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
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

        path = run_map.path(
            parameter.split("_")[0],
            f"num_measures_over_distance_{parameter}_{seed}.png",
        )
        suptitle = f"Number of measures over distance (time bin= 10.0s / dist bin = 50.0m) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)

    def plot_map_size_over_distance_grid(
        self,
        run_map: RunMap,
        app: str = "e_map",
        bin_size: float = 10.0,
        parameter="insertionOrder_ttl15",
        seed=0,
    ):
        sim: Simulation = run_map[parameter][seed]
        e_cfg: DpmmCfgDb = DpmmCfgBuilder.load_entropy_cfg(sim.data_root)
        m = MapSizeAndAgeOverDistance(
            hdf_path=e_cfg.map_size_and_age_over_distance.path
        )

        grid_figure_iter: GridPlotIter = self.create_figure()
        flierprops = dict(
            marker="2",
            markerfacecolor="gray",
            markeredgecolor="gray",
            markersize=1,
            linestyle="none",
        )
        medianprops = dict(linestyle="-", linewidth=2.5, color="firebrick")
        meanprops = dict(linestyle="dotted", linewidth=2.5, color="green")

        for idx, (_, ax, color) in enumerate(grid_figure_iter):
            rsd = idx + 1
            data = m.hdf_rsd.select(where=f"rsd={rsd} and metric=0", columns=["count"])
            stats = (
                data.groupby(["dist_left"])
                .aggregate(["mean", "min", "max", "count", calc_box_stats()])
                .droplevel(0, axis=1)
            )
            boxes = []
            for idx, df in stats.iterrows():
                boxes.append(
                    box_stats_to_plt_box(
                        label=idx, stat=df["box_stats"], mean=df["mean"]
                    )
                )

            ax.bxp(
                boxes,
                showmeans=True,
                meanline=True,
                flierprops=flierprops,
                medianprops=medianprops,
                meanprops=meanprops,
            )
            lbs = ax.xaxis.get_major_formatter().seq
            for i in range(len(lbs)):
                if lbs[i] % 500 != 0:
                    lbs[i] = ""
            ax.xaxis.set_major_formatter(FixedFormatter(lbs))
            median_line = Line2D([0], [0], label="median", color="firebrick")
            mean_line = Line2D(
                [0], [0], linestyle="dotted", label="mean", color="green"
            )
            fliers = Line2D(
                [0],
                [0],
                label="fliers",
                marker="2",
                markersize=1,
                markeredgecolor="gray",
                markerfacecolor="gray",
                linestyle="",
            )
            ax.legend(handles=[median_line, mean_line, fliers], loc="upper right")

        grid_figure_iter.set_ylabel("Number of cells in map")
        grid_figure_iter.set_xlabel("Distance of cell and measurement owner in meter)")

        path = run_map.path(
            parameter.split("_")[0], f"num_cels_over_distance_{parameter}_{seed}.png"
        )
        suptitle = f"Number of cells in distance interval (time bin= 1.0s / dist bin = 50.0m) - {parameter}"
        self.save_and_close_grid_figure(grid_figure_iter, suptitle, path)
