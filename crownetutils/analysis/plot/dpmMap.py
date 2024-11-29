"""Decentralized pedestrian measurement map (DPMM) metrics. (Density or entropy alike)"""
from __future__ import annotations

import itertools
import os
import sys
from io import StringIO
from typing import Any, List, Tuple

import matplotlib.patches as pltPatch
import matplotlib.path as pltPath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator
from omnetinireader.config_parser import ObjectValue
from pandas import IndexSlice as _i
from scipy.stats import kstest, mannwhitneyu
from shapely.geometry import Point, Polygon

import crownetutils.omnetpp.scave as Scave
from crownetutils.analysis.common import RunMap, Simulation
from crownetutils.analysis.dpmm.builder import DpmmHdfBuilder
from crownetutils.analysis.dpmm.dpmm import DpmMap
from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmKey
from crownetutils.analysis.hdf_providers.map_error_data import (
    CellCountError,
    MapCountError,
)
from crownetutils.analysis.hdf_providers.node_position import (
    CoordinateType,
    NodePositionWithRsdHdf,
)
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.omnetpp.scave import CrownetSql
from crownetutils.utils.dataframe import FrameConsumer, index_or_col
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.misc import DataSource
from crownetutils.utils.plot import (
    FigureSaver,
    FigureSaverPdfPages,
    FigureSaverSimple,
    PlotUtil_,
    Style,
    enb_with_hex,
    hex_patch,
    savefigure,
    with_axis,
)


class _PlotDpmMap(PlotUtil_):
    """Default and reusable plots for
    Decentralized Pedestrian Measurement Map (DPMM)
    """

    def __init__(self) -> None:
        super().__init__()

    @timing
    def create_common_plots_density(
        self,
        data_root: str,
        builder: DpmmHdfBuilder,
        sql: Scave.CrownetSql,
        selection: str | None = None,
        saver: FigureSaver | None = None,
    ):
        """Deprecated."""

        saver = FigureSaver.FIG(saver, FigureSaverSimple(data_root))

        selection = builder.get_selected_alg() if selection is None else selection
        dmap = builder.build_dcdMap(selection=selection)

        dmap.plot_map_count_diff(savefig=saver.with_name("dpmm"))
        msce = dmap.cell_count_measure(columns=["cell_mse"]).reset_index()
        # msce time series
        self.plot_msce_ts(msce, savefig=saver.with_name("msce_ts"))
        # msce ecdf
        self.plot_msce_ecdf(msce["cell_mse"], savefig=saver.with_name("msce_ecdf"))

    def plot_map_count_diff_by_rsd(
        self, data: MapCountError, rsd_list, saver: FigureSaver
    ):
        """Create count plot for density map for cells in *one* RSD

        Args:
            saver (FigureSaver): figer saving strategy
            nodes which are in the current RSD at the time of the measurement. Defaults to False.
        """

        def max_col(data, col_s):
            cols = [c for c in data.columns if col_s in c]
            return data[cols].max().max()

        name = saver.next_name
        font_dict = self.style.font_dict
        for rsd in rsd_list:
            data_rsd = (
                data.hdf_map_measure_rsd.select(
                    where=f"{DpmmKey.RSD_ID}={rsd}",
                    columns=["map_median_count", "map_count_p25", "map_count_p75"],
                )
                .dropna()
                .reset_index()
            )
            glb_rsd = (
                data.hdf_map_measure_rsd.select(
                    where=f"{DpmmKey.RSD_ID}={rsd}", columns=["map_glb_count"]
                )
                .dropna()
                .reset_index()
            )
            fig, ax = self._plot_count_diff(glb=glb_rsd, data=data_rsd)
            ax.set_title(
                f"Node Count over Time RSD {rsd} (with data from all domains)",
                **font_dict["title"],
            )
            ax.set_xlabel("Time [s]", **font_dict["xlabel"])
            ax.set_ylabel("Pedestrian Count", **font_dict["ylabel"])

            data_rsd_local = (
                data.hdf_map_measure_rsd_local.select(
                    where=f"{DpmmKey.RSD_ID}={rsd}",
                    columns=["map_median_count", "map_count_p25", "map_count_p75"],
                )
                .dropna()
                .reset_index()
            )
            glb_rsd_local = (
                data.hdf_map_measure_rsd_local.select(
                    where=f"{DpmmKey.RSD_ID}={rsd}", columns=["map_glb_count"]
                )
                .dropna()
                .reset_index()
            )

            fig_loc, ax_loc = self._plot_count_diff(
                glb=glb_rsd_local, data=data_rsd_local
            )
            ax_loc.set_title(
                f"Node Count over Time RSD {rsd} (with rsd local data)",
                **font_dict["title"],
            )
            ax_loc.set_xlabel("Time [s]", **font_dict["xlabel"])
            ax_loc.set_ylabel("Pedestrian Count", **font_dict["ylabel"])

            _max = np.array(
                [
                    max_col(d, "map")
                    for d in [data_rsd, data_rsd_local, glb_rsd, glb_rsd_local]
                ]
            )
            for a in [ax, ax_loc]:
                a.set_ylim(0, np.floor((_max.max() + 10) / 10) * 10)
                a.set_xlim(-10, np.floor((glb_rsd["simtime"].max() + 10) / 10) * 10)
                self.auto_major_minor_locator(a)
                if self.style.create_legend:
                    a.legend()

            saver.with_name(name).with_suffix(f"_rsd_{rsd}")(fig)
            saver.with_name(name).with_suffix(f"_rsd_local_{rsd}")(fig_loc)

    @savefigure
    @with_axis
    def plot_map_count_diff(
        self, data: MapCountError, *, ax=None
    ) -> Tuple[Figure, Axes]:
        nodes = data.hdf_map_measure
        n = (
            nodes.select(columns=["map_median_count", "map_count_p25", "map_count_p75"])
            .dropna()
            .reset_index()
        )
        glb = nodes.select(columns=["map_glb_count"]).dropna().reset_index()
        fig, ax = self._plot_count_diff(glb, data=n, ax=ax)

        font_dict = self.style.font_dict
        ax.set_title("Node Count over Time", **font_dict["title"])
        ax.set_xlabel("Time [s]", **font_dict["xlabel"])
        ax.set_ylabel("Pedestrian Count", **font_dict["ylabel"])
        if self.style.create_legend:
            ax.legend()
        self.auto_major_minor_locator(ax)
        return fig, ax

    @with_axis
    def _plot_count_diff(self, glb, data, *, ax=None) -> Tuple[Figure, Axes]:
        ax.plot("simtime", "map_median_count", data=data, label="Median count")
        ax.fill_between(
            data["simtime"],
            data["map_count_p25"],
            data["map_count_p75"],
            alpha=0.35,
            interpolate=True,
            label="[Q1;Q3]",
        )
        ax.plot("simtime", "map_glb_count", data=glb, label="Actual count")
        return ax.get_figure(), ax

    def plot_msce_ts_rsd_figure_per_rsd(
        self,
        data: CellCountError,
        rsd_list: List[int],
        saver: FigureSaverSimple,
        local_only: bool,
    ):
        """Generate stand alone figures for MSCE time series

        Args:
            data (CellCountError): Data provider for MSCE data
            rsd_list (List[int]): list of RSD's to plot single figures
            saver (FigureSaverSimple): Callable object to save the figure
            inner_view (bool): If true only use data for nodes that are part of the selected rsd.
                                Otherwise use all data.
        """
        saver_name = saver.peek_next_name
        for rsd in rsd_list:
            if local_only:
                df = data.hdf_cell_measure_rsd_local.select(
                    where=f"rsd_id={rsd}", columns=["cell_mse"]
                )
                suffix = f"_rsd_local_{rsd}"
            else:
                df = data.hdf_cell_measure_rsd.select(
                    where=f"rsd_id={rsd}", columns=["cell_mse"]
                )
                suffix = f"_rsd_{rsd}"
            fig, _ = self.plot_msce_ts(data=df)
            saver.with_name(saver_name).with_suffix(suffix)(fig)
            plt.close(fig)

    @with_axis
    @savefigure
    def plot_msce_ts(
        self,
        data: pd.DataFrame,
        x="simtime",
        y="cell_mse",
        *,
        ax: plt.Axes | None = None,
    ):
        ax.scatter(
            index_or_col(data, x),
            index_or_col(data, y),
            s=int(self.par("lines.markersize", 6) / 2),
            alpha=0.5,
            label="Mean squared cell error",
        )
        ax.set_title("Mean squared cell error (MSCE) over time")
        ax.set_ylabel("Mean squared cell error (MSCE)")
        ax.set_xlabel("Simulation time in seconds")
        self.auto_major_minor_locator(ax)
        ax.legend()
        return ax.get_figure(), ax

    @with_axis
    @savefigure
    def cmp_msce_ecdf(
        self,
        sims: List[Simulation],
        *,
        frame_c: FrameConsumer = FrameConsumer.EMPTY,
        ax: plt.Axes | None = None,
    ):
        for sim in sims:
            dmap = sim.get_dcdMap()
            msce = dmap.cell_count_measure(columns=["cell_mse"])
            # apply frame consumer if present
            msce = frame_c(msce).reset_index()
            print(sim.label)
            print(msce["cell_mse"].describe())
            self.plot_ecdf(msce["cell_mse"], label=sim.label, ax=ax)

        ax.legend()
        ax.set_xlabel("MSCE")
        ax.set_title("ECDF: Mean squared cell error (MSCE) comparison")
        self.auto_major_minor_locator(ax)
        return ax.get_figure(), ax

    def plot_msce_ecdf_rsd_figure_per_rsd(
        self,
        data: CellCountError,
        rsd_list: List[int],
        saver: FigureSaverSimple,
        inner_view: bool,
    ):
        """Generate stand alone figures for MSCE CDF

        Args:
            data (CellCountError): Data provider for MSCE data
            rsd_list (List[int]): list of RSD's to plot single figures
            saver (FigureSaverSimple): Callable object to save the figure
            inner_view (bool): If true only use data for nodes that are part of the selected rsd.
                                Otherwise use all data.
        """
        saver_name = saver.peek_next_name
        for rsd in rsd_list:
            if inner_view:
                df = data.hdf_cell_measure_rsd_local.select(
                    where=f"rsd_id={rsd}", columns=["cell_mse"]
                )
                suffix = f"_rsd_local_{rsd}"
            else:
                df = data.hdf_cell_measure_rsd.select(
                    where=f"rsd_id={rsd}", columns=["cell_mse"]
                )
                suffix = f"_rsd_{rsd}"
            fig, _ = self.plot_msce_ecdf(df["cell_mse"])
            saver.with_name(saver_name).with_suffix(suffix)(fig)
            plt.close(fig)

    @with_axis
    @savefigure
    def plot_msce_ecdf(self, data, *, ax: plt.Axes | None = None):
        ax = self.plot_ecdf(data, label="MSCE", ax=ax)
        ax.set_title("ECDF: Mean squared cell error (MSCE)")
        ax.set_xlabel("MSCE")
        ax.legend()
        self.auto_major_minor_locator(ax)
        return ax.get_figure(), ax

    @timing
    def plot_map_pkt_count_all(
        self,
        data_root: str,
        sql: CrownetSql,
        saver: FigureSaver | None = None,
    ):
        saver = FigureSaver.FIG(saver)
        data = self.get_map_pkt_count_ts(sql)
        fig, ax = plt.subplots()
        self.df_to_table(data.describe().applymap("{:1.4f}".format).reset_index(), ax)
        ax.set_title(f"Descriptive statistics for map application")
        saver(fig, os.path.join(data_root, f"tx_MapPkt_stat.pdf"))

        fig, ax = self.check_ax()
        ax.scatter("time", "pkt_count", data=data.reset_index())
        ax.set_title("Packet count over time")
        ax.set_ylabel("Number of packets")
        ax.set_xlabel("Simulation time in seconds")
        saver(fig, os.path.join(data_root, f"txMapPktCount_ts.pdf"))

    @timing
    @with_axis
    @savefigure
    def create_plot_err_box_over_time(
        self, sim: Simulation, title: str, *, ax: plt.Axes | None = None
    ):
        s = Style()
        s.font_dict = {
            "title": {"fontsize": 14},
            "xlabel": {"fontsize": 10},
            "ylabel": {"fontsize": 10},
            "legend": {"size": 14},
            "tick_size": 10,
        }
        s.create_legend = False

        dmap = sim.get_dcdMap()
        dmap.style = s
        fig, ax = dmap.plot_err_box_over_time(ax=ax, xtick_sep=10)
        ax.set_title(title)
        return fig, ax

    def err_hist_plot(self, s: Style, data: List[DataSource]):
        """DPMM plot for Dash-Tool"""

        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_error_histogram(ax=axes[idx], data_source=run)
            a.set_title(title(sim))

        self.equalize_axis(fig.get_axes(), "y")
        self.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Count Error Histogram")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        return fig

    def diff_plot(self, s: Style, data: List[DataSource]):
        """DPMM plot for Dash-Tool

        Args:
            s (Style): _description_
            data (List[DataSource]): _description_
        """

        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_map_count_diff(ax=axes[idx], data_source=run)
            a.set_title(title(sim))

        # fix legends
        x = axes[0].legend()
        axes[0].get_legend().remove()
        self.equalize_axis(fig.get_axes(), "y")
        self.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Comparing Map count with ground truth over time")
        fig.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
        fig.legend(
            x.legendHandles,
            [i._text for i in x.texts],
            ncol=3,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.0),
        )

        return fig

    def err_box_plot(self, s: Style, data: List[DataSource]):
        """DPMM plot for Dash-Tool

        Args:
            s (Style): _description_
            data (List[DataSource]): _description_
        """

        def title(sim: Simulation):
            cfg = sim.run_context.oppini
            map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
            run_name = sim.run_context.args.get_value("--run-name")
            if all(i in map for i in ["alpha", "stepDist", "zeroStep"]):
                return f"{run_name[0:-7]}\nalpha:{map['alpha']} distance threshold: {map['stepDist']} use zero: {map['zeroStep']}"
            else:
                return f"{run_name[0:-7]}\n Youngest measurement first"

        fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(16, 9))

        axes = [a for aa in ax for a in aa]
        for a in axes[len(data) :]:
            a.remove()

        for idx, run in enumerate(data):
            sim: Simulation = run.source
            dmap = sim.get_dcdMap()
            dmap.style = s
            # provide data from run cache
            _, a = dmap.plot_err_box_over_time(
                ax=axes[idx], xtick_sep=10, data_source=run
            )
            a.set_title(title(sim))

        self.equalize_axis(fig.get_axes(), "y")
        self.equalize_axis(fig.get_axes(), "x")
        fig.suptitle("Count Error over time")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 1.0))

        return fig

    def box_plot(self, data: pd.DataFrame, bin_width, bin_key):
        """DPMM plot for Dash-Tool

        Args:
            data (pd.DataFrame): _description_
            bin_width (_type_): _description_
            bin_key (_type_): _description_

        Returns:
            _type_: _description_
        """

        if bin_key in data.columns:
            data = data.set_index(bin_key, verify_integrity=False)

        bins = int(np.floor(data.index.max() / bin_width))
        _cut = pd.cut(data.index, bins)
        return data.groupby(_cut), _cut

    def plot_descriptive_comparison(
        self,
        data: pd.DataFrame,
        lbl_dict: dict,
        run_map: RunMap,
        out_name: str,
        stat_col_combination: List[Tuple[Any, Any]] | None = None,
        pdf_file=None,
        palette=None,
        value_axes_label: str = "value",
    ):
        """Save mulitple descriptive plots and statisitcs based on given data.
        DataFrame must be in the long format with a single index.
        """

        if pdf_file is None:
            with run_map.pdf_page(out_name) as pdf:
                self.plot_descriptive_comparison(
                    data,
                    lbl_dict,
                    run_map,
                    out_name,
                    stat_col_combination,
                    pdf,
                    palette,
                )
        else:
            if data.shape[1] <= 3:
                f, (stat_ax, descr_ax) = plt.subplots(2, 1, figsize=(16, 9))
                f = [f]
            else:
                f_1, stat_ax = self.check_ax()
                f_2, descr_ax = self.check_ax()
                f = [f_1, f_2]

            self.calculate_equality_tests(
                data,
                combination=stat_col_combination,
                lbl_dict=lbl_dict,
                ax=stat_ax,
                path=run_map.path(out_name.replace(".pdf", "_stats.csv")),
            )

            descr_ax.set_title("Summary Statistics")
            df = data.describe().applymap("{:.6f}".format).reset_index()
            df.to_csv(run_map.path(out_name.replace(".pdf", "_summary.csv")))
            descr_ax.axis("off")
            tbl = descr_ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.scale(1, 2)

            for _f in f:
                _f.tight_layout()
                pdf_file.savefig(_f)
                plt.close(_f)

            # Line plot
            f, ax = self.check_ax()
            sns.lineplot(data=data, ax=ax, palette=palette)
            ax.set_title(f"Time Series")
            ax.set_xlabel("Time in seconds")
            ax.set_ylabel(value_axes_label)
            self.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            # ECDF plot
            f, ax = self.check_ax()
            sns.ecdfplot(data, ax=ax, palette=palette)
            ax.set_title(f"ECDF pedestrian count")
            ax.set_xlabel(value_axes_label)
            ax.get_legend().set_title(None)
            sns.move_legend(ax, "upper left")
            self.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            # Hist plot
            f, ax = self.check_ax()
            sns.histplot(
                data,
                cumulative=False,
                common_norm=False,
                stat="percent",
                element="step",
                ax=ax,
                palette=palette,
            )
            ax.set_title(f"Histogram of pedestrian count")
            ax.set_xlabel(value_axes_label)
            ax.get_legend().set_title(None)
            sns.move_legend(ax, "upper left")
            self.rename_legend(ax, rename=lbl_dict)
            pdf_file.savefig(f)
            plt.close(f)

            if data.shape[1] <= 3:
                f, (box, violin) = plt.subplots(1, 2, figsize=(16, 9))
                f = [f]
            else:
                f_box, box = self.check_ax()
                f_violin, violin = self.check_ax()
                f = [f_box, f_violin]

            # Box plot
            sns.boxplot(data=data, ax=box, palette=palette)
            box.set_title(f"Boxplot of pedestrian count")
            box.set_xlabel("Data")
            box.set_ylabel(value_axes_label)

            # Violin plot
            sns.violinplot(data=data, ax=violin, palette=palette)
            violin.set_title(f"Violin of pedestrian count")
            violin.set_xlabel("Data")
            violin.set_ylabel(value_axes_label)
            for _f in f:
                pdf_file.savefig(_f)
                plt.close(_f)

    def calculate_equality_tests(
        self,
        data: pd.DataFrame,
        combination: list[Tuple[Any, Any]] | None = None,
        lbl_dict: dict | None = None,
        ax: plt.Axes | None = None,
        path: str | None = None,
    ):
        lbl_dict = {} if lbl_dict is None else lbl_dict
        if combination is None:
            combination = list(itertools.combinations(data.columns, 2))
        res = []
        for left, right in combination:
            _s = {}
            _mw = mannwhitneyu(data[left], data[right], method="asymptotic")
            _ks = kstest(data[left], data[right])
            _s["pair"] = f"{lbl_dict.get(left, left)} - {lbl_dict.get(right, right)}"
            _s["mw_stat"] = _mw.statistic
            _s["mw_p"] = _mw.pvalue
            _s["mw_H"] = "$H_0$ (same)" if _s["mw_p"] > 0.05 else "$H_1$ (different)"
            _s["ks_stat"] = _ks.statistic
            _s["ks_p"] = _ks.pvalue
            _s["ks_H"] = "$H_0$ (same)" if _s["ks_p"] > 0.05 else "$H_1$ (different)"
            res.append(_s)
        df = pd.DataFrame.from_records(res, index="pair")
        df.update(df[["mw_stat", "mw_p", "ks_stat", "ks_p"]].applymap("{:.6e}".format))
        df = df.reset_index()

        if path is not None:
            df.to_csv(path)

        if ax is None:
            return df
        else:
            ax.set_title("Test for similarity of distribution")
            ax.axis("off")
            tbl = ax.table(cellText=df.values, colLabels=df.columns, loc="center")
            tbl.scale(1, 2)
            ax.get_figure().tight_layout()
            return df, ax

    @savefigure
    @with_axis
    def cmp_plot__map_count_diff(
        self, sims: List[Simulation], ax: plt.Axes | None = None
    ):
        for sim in sims:
            nodes = sim.get_dcdMap().map_count_measure()
            df = (
                nodes.loc[:, ["map_mean_count", "map_count_p25", "map_count_p75"]]
                .dropna()
                .reset_index()
            )
            self.fill_between(
                data=df,
                x="simtime",
                val="map_mean_count",
                fill_val=["map_count_p25", "map_count_p75"],
                plot_lbl=f"{sim.label}: mean count",
                fill_args=dict(label=f"{sim.label}: Q1;Q3"),
                ax=ax,
            )
            _g = nodes.loc[:, ["map_glb_count"]].dropna().reset_index()
            ax.plot(
                "simtime", "map_glb_count", data=_g, label=f"{sim.label}: actual count"
            )

        ax.set_title("Node count over time")
        ax.set_ylabel("Pedestrian count")
        ax.set_xlabel("Simulation time in seconds")
        ax.legend()
        return ax.get_figure(), ax

    def pull_error_data(
        self,
        measure: pd.Series,
        dpmm: DpmMap,
        node_pos: NodePositionWithRsdHdf,
        coord: CoordinateType,
        out: StringIO,
        delta_t: float = 10.0,
    ) -> [plt.Figure, plt.Axes]:
        ue = (
            node_pos.ue.select(
                where=f"hostId={measure['ID']} and time>={measure['simtime']-delta_t} and time<={measure['simtime']+5}"
            )
            .set_index(["time"])
            .sort_index()
        )
        x = measure["x"]
        y = measure["y"]
        entry_log = dpmm._map_p.select(
            where=f"ID={measure['ID']} and x={x} and y={y} and simtime>={measure['simtime']-delta_t} and simtime<={measure['simtime']+5}",
            columns=[
                "count",
                "measured_t",
                "received_t",
                "selection",
                "selectionRank",
                "rsd_id",
                "owner_rsd_id",
                "delay",
                "measurement_age",
                "update_age",
            ],
        )

        colors = node_pos.enb_colors()
        with MapPlotter(node_pos=node_pos, coord=coord, with_icon=True) as (
            map_plotter
        ):
            map_plotter.ax.scatter(
                ue[coord.x],
                ue[coord.y],
                c=ue["servingEnb"].astype(int).apply(lambda x: colors[x]),
                marker=".",
            )
            map_plotter.ax.scatter(
                measure["x"],
                measure["y"],
                color="red",
                marker=".",
                alpha=1.0,
                label=f"wrong RSD assignement by {measure['ID']}",
            )

        out.write("Error entry:\n")
        out.write(str(measure))
        out.write("\n\nPosition trace:\n")
        out.write(str(ue))
        out.write("\n")
        entry_log["error"] = ""
        entry_log.loc[measure["simtime"], "error"] = "<<<"
        out.write(str(entry_log))
        out.write("\n")
        out.write("#" * 120)
        out.write("\n")

        return map_plotter

    def plot_ground_truth_tiles(
        self,
        sim: Simulation,
        node_pos: NodePositionWithRsdHdf,
        coord: CoordinateType = CoordinateType.xy_cell,
        rsd: int = 1,
    ):
        # enb bounds
        colors = node_pos.enb_colors()
        enb = node_pos.enb.frame()
        ue = node_pos.ue.select(where=f"time=200")
        ue_rsd_colors = ue["servingEnb"].astype(int).apply(lambda x: colors[x])

        # rsd_1_hex =
        outter_r = 650.0 / (np.sqrt(3) / 2) + 150  # + Randbereich
        xy = []
        for i in range(6):
            xy.append(
                [outter_r * np.cos(i * np.pi / 3), outter_r * np.sin(i * np.pi / 3)]
            )
        xy = np.array(xy)
        xy = xy + enb[coord.cols].values[0]
        rsd_1_hex: Polygon = Polygon(xy)

        # ground truth cells
        dpmm = sim.get_dcdMap()

        data = pd.read_csv("rsd_1_with_error_marker_dzone150.csv")
        # data = dpmm._map_p.select(
        #         where="rsd_id=1 and simtime <= 200",
        #     ).reset_index()
        # data["in_rsd"] = data.apply(lambda x: rsd_1_hex.contains(Point(x["x"], x["y"])) , axis=1)
        # data = data.sort_values(["simtime", "x", "y"])
        # data.to_csv("rsd_1_with_error_marker_dzone150.csv")

        wrong = data[~data["in_rsd"]].sort_values(
            [
                "simtime",
                "ID",
                "x",
                "y",
            ]
        )
        wrong_own_m = wrong[wrong["ID"] == wrong["source"]]
        wrong_foreign = wrong[wrong["ID"] != wrong["source"]]
        u = (
            node_pos.ue.select(where="hostId=7768 and time <= 58.0")
            .set_index(["time"])
            .sort_index()
        )

        err_count = wrong.shape[0]
        path = f"/mnt/data1tb/results/arc-dsa_multi_cell/s2_ttl_and_stream/simulation_runs/outputs/Sample_0_0/final_multi_enb_out/density/fig_out/err_trace.txt"
        with open(path, "w", encoding="utf-8") as fd:
            for row_id, (_, r) in enumerate(wrong.iterrows()):
                print(f"{row_id}/{err_count}", file=sys.stdout)
                print(f"{row_id}/{err_count}", file=fd)
                plotter: MapPlotter = self.pull_error_data(
                    r, dpmm=dpmm, node_pos=node_pos, coord=coord, out=fd
                )
                plotter.ax.set_title(
                    f"Node {r.ID} at time {r.simtime} wrong RSD association for cell [{r.x}, {r.y}]"
                )
                path = f"/mnt/data1tb/results/arc-dsa_multi_cell/s2_ttl_and_stream/simulation_runs/outputs/Sample_0_0/final_multi_enb_out/density/fig_out/err_trace_{row_id}of{err_count}.png"
                plotter.save_and_close(path)
                print("", file=fd)
        print("done.")
        # fig, ax = self.check_ax()
        # dfc = data[(data["simtime"] == 200) & data["in_rsd"]]
        # dfi = data[(data["simtime"] == 200) & ~data["in_rsd"]]
        # # ax.scatter(dfc["x"], dfc["y"], color="green", marker=".", alpha=.4, label="map data correctly labeled")
        # # ax.scatter(dfi["x"], dfi["y"], color="red", marker=".", alpha=.4, label="map data incorrectly labeled")
        # # ax.scatter(ue[coord.x], ue[coord.y], c=ue_rsd_colors, alpha=1., marker=".", label="ue position data")
        # ax.scatter(u[coord.x], u[coord.y], c=u["servingEnb"].astype(int).apply(lambda x: colors[x]), marker=".")
        # ax.scatter(wrong["x"].values[0], wrong["y"].values[0], color="red", marker=".", alpha=.5)

        # # append enb hex
        # enb_patches, hex_patches = enb_with_hex(
        #     origin=enb[coord.cols].values,
        #     inner_r=650,
        #     scale=100)
        # ax.add_collection(hex_patches)
        # # ax.add_collection(enb_patches)
        # ax.add_collection(
        #     PatchCollection(
        #         [hex_patch(origin=enb[coord.cols].values[0], outter_r=outter_r)],
        #         facecolors="none",
        #         edgecolors="green",
        #     )
        # )

        # if coord.is_cartesian:
        #     ax.xaxis.set_major_locator(MultipleLocator(500))
        #     ax.yaxis.set_major_locator(MultipleLocator(500))
        #     ax.xaxis.set_minor_locator(MultipleLocator(100))
        #     ax.yaxis.set_minor_locator(MultipleLocator(100))

        #     _min = (0, 500)
        #     _max = (6300, 5300)
        #     ax.set_xlim(_min[0], _max[0])
        #     ax.set_ylim(_min[1], _max[1])
        #     ax.set_aspect("equal")

        # ax.legend()
        # fig.tight_layout()
        # fig.savefig("/mnt/data1tb/results/arc-dsa_multi_cell/s2_ttl_and_stream/simulation_runs/outputs/Sample_0_0/final_multi_enb_out/density/fig_out/trace_TEST.png")


class MapPlotter(PlotUtil_):
    def __init__(
        self, node_pos, coord, with_icon: bool = False, hex_inner_r=650
    ) -> None:
        super().__init__()
        self.node_pos: NodePositionWithRsdHdf = node_pos
        self.coord: CoordinateType = coord
        self.ax = None
        self.fig: plt.figure = None
        self.with_icon: bool = with_icon
        self.hex_inner_r = hex_inner_r

    def __call__(self, ax: plt.Axes = None) -> MapPlotter:
        self.ax = ax
        return self

    def add_enb_patches(self, ax: plt.Axes, zorder=1):
        enb = self.node_pos.enb.frame()
        enb_patches, hex_patches = enb_with_hex(
            origin=enb[self.coord.cols].values, inner_r=self.hex_inner_r, scale=100
        )
        ax.add_collection(hex_patches)

    def __enter__(self) -> MapPlotter:
        f, a = self.check_ax(self.ax)
        self.ax = a
        self.fig = f

        enb = self.node_pos.enb.frame()
        enb_patches, hex_patches = enb_with_hex(
            origin=enb[self.coord.cols].values, inner_r=self.hex_inner_r, scale=100
        )
        self.ax.add_collection(hex_patches)
        self.enb_patches = enb_patches

        return self

    def __exit__(self, _t, value, tb):
        if self.with_icon:
            self.ax.add_collection(self.enb_patches)

        if self.coord.is_cartesian:
            self.ax.xaxis.set_major_locator(MultipleLocator(500))
            self.ax.yaxis.set_major_locator(MultipleLocator(500))
            self.ax.xaxis.set_minor_locator(MultipleLocator(100))
            self.ax.yaxis.set_minor_locator(MultipleLocator(100))
            self.ax.set_ylabel("North in meter")
            self.ax.set_xlabel("East in meter")

            _min = (0, 500)
            _max = (6300, 5300)
            self.ax.set_xlim(_min[0], _max[0])
            self.ax.set_ylim(_min[1], _max[1])
            self.ax.set_aspect("equal")
        self.ax.legend(loc="lower right")

    def set_xy_limit(self, ax: plt.Axes, xy, offset=100):
        if isinstance(xy, pd.DataFrame):
            xy = xy[self.coord.col]
        _max = np.max(xy, axis=0)
        _min = np.min(xy, axis=0)
        ax.set_xlim((_min[0] - offset, _max[0] + offset))
        ax.set_ylim(_min[1] - offset, _max[1] + offset)

    def add_cell_patches(
        self, ax: plt.Axes, cells, size, facecolor="none", edgecolors="gray", zorder=1
    ):
        patches = []
        if isinstance(cells, pd.DataFrame):
            cells = (
                pd.MultiIndex(cells[self.coord.cols], names=["x", "y"])
                .unique()
                .to_frame()
                .values
            )
        for cell in cells:
            patches.append(pltPatch.Rectangle(cell, width=size, height=size))

        cell_patches = PatchCollection(
            patches=patches, facecolor=facecolor, edgecolors=edgecolors, zorder=zorder
        )
        ax.add_collection(cell_patches)

    def save_and_close(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)
        plt.close(self.fig)
        self.fig = None
        self.ax = None


PlotDpmMap = _PlotDpmMap()
