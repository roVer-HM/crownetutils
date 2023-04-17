from __future__ import annotations

import itertools
import os
from typing import Any, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from omnetinireader.config_parser import ObjectValue
from scipy.stats import kstest, mannwhitneyu

import crownetutils.omnetpp.scave as Scave
from crownetutils.analysis.common import RunMap, Simulation
from crownetutils.analysis.dpmm.builder import DpmmHdfBuilder
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.utils.dataframe import FrameConsumer
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.misc import DataSource
from crownetutils.utils.plot import (
    FigureSaver,
    FigureSaverPdfPages,
    PlotUtil_,
    Style,
    savefigure,
    with_axis,
)


class _PlotDpmMap(PlotUtil_):
    """Default and reusable plots for
    Decentralized Pedestrian Measurement Map (DPMM)
    """

    @timing
    def create_common_plots_density(
        self,
        data_root: str,
        builder: DpmmHdfBuilder,
        sql: Scave.CrownetSql,
        selection: str | None = None,
    ):
        out_dir = os.path.join(data_root, "common_output.pdf")
        selection = builder.get_selected_alg() if selection is None else selection
        dmap = builder.build_dcdMap(selection=selection)
        with PdfPages(out_dir) as pdf:
            dmap.plot_map_count_diff(savefig=pdf)
            msce = dmap.cell_count_measure(columns=["cell_mse"]).reset_index()
            # msce time series
            self.plot_msce_ts(msce, savefig=pdf)
            # msce ecdf
            self.plot_msce_ecdf(msce["cell_mse"], savefig=pdf)

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
            x,
            y,
            data=data,
            s=int(self.par("lines.markersize", 6) / 2),
            alpha=0.5,
            label="Mean squared cell error",
        )
        ax.set_title("Mean squared cell error (MSCE) over time")
        ax.set_ylabel("Mean squared cell error (MSCE)")
        ax.set_xlabel("Simulation time in seconds")
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
            self.ecdf(msce["cell_mse"], label=sim.label, ax=ax)

        ax.legend()
        ax.set_xlabel("MSCE")
        ax.set_title("ECDF: Mean squared cell error (MSCE) comparison")
        return ax.get_figure(), ax

    @with_axis
    @savefigure
    def plot_msce_ecdf(self, data, *, ax: plt.Axes | None = None):
        ax = self.ecdf(data, label="MSCE")
        ax.set_title("ECDF: Mean squared cell error (MSCE)")
        ax.set_xlabel("MSCE")
        ax.legend()
        return ax.get_figure(), ax

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


PlotDpmMap = _PlotDpmMap()
