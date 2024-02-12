from __future__ import annotations

import os
from itertools import product
from typing import List

import numpy as np
import pandas as pd

from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.analysis.hdf_providers.helper import ExpectedHdfContent
from crownetutils.analysis.hdf_providers.node_position import NodePositionWithRsdHdf
from crownetutils.analysis.hdf_providers.sql_app_proxy import SqlAppProxy
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.utils.logging import LogWriter, logger
from crownetutils.utils.misc import Timer


class NodeTxData:
    base_groups = ["tx_bytes", "tx_burst", "tx_interval"]

    ATTR_rsd = "with_rsd"
    ATTR_max_bw = "max_application_bandwidth_in_bps"

    def __init__(
        self,
        hdf_path: str,
        apps: List[SqlAppProxy],
    ) -> None:
        self.apps: List[SqlAppProxy] = apps
        self.hdf_path = hdf_path
        self._hdf: BaseHdfProvider = None

        self.groups = []
        for app in self.apps:
            for base_g in self.base_groups:
                self.groups.append(self.g(app, path=base_g))

    def g(self, app: str | SqlAppProxy, path: str) -> str:
        if isinstance(app, SqlAppProxy):
            return app.group_by_app(path)
        else:
            return f"{app}/{path}"

    @property
    def hdf(self) -> BaseHdfProvider:
        if self._hdf is None:
            self._hdf = BaseHdfProvider(self.hdf_path)
        return self._hdf

    def frame_by_app(
        self,
        data: str,
        apps: SqlAppProxy | str | List[SqlAppProxy | str] = None,
        where: str = None,
        columns: List[str] = None,
        append_application_column: bool = True,
    ):
        """Return filtered data based on 'group' and concatenate data for provided applications. If none given all applications are included.

        Args:
            data (str): data name (tx_bytes, tx_burst or tx_interval)
            apps (SqlAppProxy | str | List[SqlAppProxy | str], optional): Any root level group name. Defaults to None, meaning all root level groups are searched for 'data group name'.
            where (str, optional): Hdf query string applied to all applications. Defaults to None.
            columns (List[str], optional): Hdf column names applied to all applications. Defaults to None.
            append_application_column (bool, optional):

        Returns:
            _type_: _description_
        """
        if apps is None:
            apps = self.hdf.get_groups()
        ret = []
        for app in apps:
            _hdf = BaseHdfProvider(self.hdf_path, group=self.g(app, data))
            _df = _hdf.select(where=where, columns=columns)
            if append_application_column:
                _df["app"] = app
            ret.append(_df)

        ret = pd.concat(ret, axis=0)
        return ret

    def tx_bytes(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        """HdfProvider of the form [hostId, time, tx_bytes, servingEnb] (no index)

        Only contains data for the provided application. No aggregation applied. For group information
        see 'print_hdf_info()'. `hostId` , `time` do not build a unique index. See tx_burst where packets
        from the same time are aggregated into messages (i.e. bursts)

        Call .frame() method for all data or use IndexSlicing [] or the .select()
        method for on disk filtering.

        Args:
            app (str | SqlAppProxy): group identifier which yields group name as shown in print_hdf_info()

        Returns:
            BaseHdfProvider: HdfProvider configured with provided application and group.
        """
        return BaseHdfProvider(self.hdf_path, group=self.g(app, "tx_bytes"))

    def tx_burst(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        """HdfProvider of the form (hostId, time)[burst_num, burst_size, servingEnb]

        Only contains data for the provided application. Contains sum-aggregated packets. For group information
        see 'print_hdf_info()'.

        `burst_num`:    number of packets contained in one message (i.e. burst)
        `burst_size`:   Total size of burst in bytes. (sum-aggregate of tx_bytes over the key hostId, time )

        Call .frame() method for all data or use IndexSlicing [] or the .select()
        method for on disk filtering.

        Args:
            app (str | SqlAppProxy): group identifier which yields group name as shown in print_hdf_info()

        Returns:
            BaseHdfProvider: HdfProvider configured with provided application and group.
        """
        return BaseHdfProvider(self.hdf_path, group=self.g(app, "tx_burst"))

    def tx_interval(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        """HdfProvider of the form (hostId, time)[tx_interval_det, tx_interval, member_count, servingEnb]

        Only contains data for the provided application. No aggregation applied. For group information
        see 'print_hdf_info()'.
        `tx_interval_det`:  Deterministic transmission interval calculated based on average message size and number
                            of competing nodes for shared resources
        `tx_interval`:      Smeared transmission interval used. (Mitigate synchronization)
        `member_count`:     Number nodes competing for resources at this time.

        Call .frame() method for all data or use IndexSlicing [] or the .select()
        method for on disk filtering.

        Args:
            app (str | SqlAppProxy): group identifier which yields group name as shown in print_hdf_info()

        Returns:
            BaseHdfProvider: HdfProvider configured with provided application and group.
        """
        return BaseHdfProvider(self.hdf_path, group=self.g(app, "tx_interval"))

    def tx_bytes_per_app(
        self, apps: str | SqlAppProxy | None = None, where_clause: str | None = None
    ) -> pd.DataFrame:
        """Creates frame of the form (app, time)[value] to work with throuput methods in OppAnalysis

        Args:
            apps (str | SqlAppProxy | None, optional): If None use group names in hdf as app identifier. Defaults to None.
            where_clause (str | None, optional): Use to filter by time, hostId or servingEnb (aka RSD). Defaults to None.

        Returns:
            pd.DataFrame: Frame of the form (app, time)[value] (sorted index)
        """
        if apps is None:
            # use all base based on top level groups
            apps = self.hdf.get_groups()
        ret = []
        for app in apps:
            _df = self.tx_bytes(app=app).select(
                where=where_clause, columns=["time", "tx_bytes"]
            )
            _df["app"] = app
            ret.append(_df)

        ret = pd.concat(ret, axis=0)
        ret = (
            ret[["app", "time", "tx_bytes"]]
            .set_axis(["app", "time", "value"], axis=1)
            .set_index(["app", "time"])
            .sort_index()
        )
        return ret

    def get_information_transfer_per_burst(
        self,
        app: str | SqlAppProxy,
        map_size_data: pd.Da.DataFrame,
        time_range_start: float = 0.0,
        time_range_end: float = -1.0,
        bin_size: float = 1.0,
        where=None,
        map_header_bytes: int = 30,
        cell_bytes: int = 12,
    ) -> pd.DataFrame:
        """Create time series of burst size combined with total map size and information transfer per burst ratio.

        Frame structure: (hostId, time)["burst_num", "burst_size", "servingEnb", "cells_per_burst", "map_size", "information_transfer_per_burst"]

        with information_transfer_per_burst = cells_per_burst / map_size

        Args:
            app (str | SqlAppProxy): Application to use
            map_size_data (pd.Da.DataFrame): frame with map size for each hostId at each time
            time_range_start (float, optional): Time range start . Defaults to 0.0.
            time_range_end (float, optional): Time range end. If < 0 it will be max("simtime") + bin_size. Defaults to -1.0.
            bin_size (float, optional): size of interval index applied on map_size_data. Defaults to 1.0.
            where (_type_, optional): HDF filter for application data. Defaults to None.
            map_header_bytes (int, optional): Size of map header for each packet. Defaults to 30.
            cell_bytes (int, optional): Size of cell encoded in map packet. Defaults to 12.

        Returns:
            pd.DataFrame:
        """

        if time_range_end < time_range_start:
            time_range_end = map_size_data["simtime"].max() + bin_size

        bins = pd.interval_range(
            start=time_range_start, end=time_range_end, freq=bin_size, closed="right"
        )
        map_size_data["time_interval"] = pd.cut(map_size_data["simtime"], bins=bins)

        burst = self.tx_burst(app=app).select(where=where)
        burst["cells_per_burst"] = (
            burst["burst_size"] - burst["burst_num"] * map_header_bytes
        ) / cell_bytes
        burst["time_interval"] = pd.cut(burst.index.get_level_values("time"), bins=bins)
        burst = burst.reset_index().set_index(["hostId", "time_interval"]).sort_index()

        burst = pd.merge(burst, map_size_data, on=["hostId", "time_interval"]).dropna()
        burst = burst.rename(columns={"numberOfCells": "map_size"})

        # total map size is only reported at 1 second intervals. The map size at
        # times between these intervals can vary. Ensure that at least
        # cell_per_burst number of cells are present.
        burst["map_size"] = (
            np.concatenate([burst["map_size"].values, burst["cells_per_burst"].values])
            .reshape((2, -1))
            .T.max(axis=1)
        )

        burst["information_transfer_per_burst"] = (
            burst["cells_per_burst"] / burst["map_size"]
        )

        burst = burst.set_index(["hostId", "time"]).sort_index()
        burst = burst.drop(columns=["simtime", "time_interval"])
        return burst

    def get_target_rates(self, bps_to_multiplier: float = 1) -> dict:
        apps = self.hdf.get_groups()
        target_rates = {
            app: self.tx_bytes(app).get_attribute(self.ATTR_max_bw) * bps_to_multiplier
            for app in apps
        }
        return target_rates

    def get_tx_throughput_diff_by_app(
        self,
        target_rates: dict = None,
        bin_size: float = 10.0,
        throughput_unit: float = 1000.0,
        serving_enb: int | None = None,
    ):
        """_summary_

        Args:
            target_rates (dict, optional): _description_. Defaults to None.
            bin_size (float, optional): _description_. Defaults to 10.0.
            throughput_unit (float, optional): _description_. Defaults to 1000.0.
            serving_enb (int | None, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if serving_enb is not None:
            serving_enb = f"servingEnb={serving_enb}"

        if target_rates is None:
            target_rates = self.get_target_rates()

        data = self.tx_bytes_per_app(
            apps=list(target_rates.keys()), where_clause=serving_enb
        )

        tx_rate = OppAnalysis.get_sent_packet_throughput_by_app(
            sql=None,
            freq=bin_size,
            tx_byte_data=data,
            hdf=None,
            throughput_unit=throughput_unit,
        )

        for c in tx_rate.columns:
            _rate = target_rates[c] / throughput_unit
            tx_rate[f"diff_{c}"] = tx_rate[c] - _rate
        return tx_rate

    @classmethod
    def get_or_create(
        cls,
        hdf_path: str,
        apps: List[SqlAppProxy],
        node_pos: NodePositionWithRsdHdf = None,
        override_existing: bool = True,
    ) -> NodeTxData:
        obj: NodeTxData = cls(hdf_path, apps)

        if os.path.exists(hdf_path):
            expected_content = ExpectedHdfContent().add_groups(
                groups=obj.groups, **{obj.ATTR_rsd: None, obj.ATTR_max_bw: None}
            )
            is_content_as_expected, diff = expected_content.test_hdf(
                BaseHdfProvider(hdf_path)
            )

            if is_content_as_expected:
                # hdf file exists and contains all data with same parameters
                logger.info(
                    f"found existing {cls.__name__} file with matching parameter setup. No build required."
                )
            else:
                logger.info("found difference in existing hdf file.")
                diff.write_diff(writer=LogWriter.info(), header=f"{cls.__name__}:")
                if not override_existing:
                    raise ValueError(
                        f"found existing {cls.__name__} file with inconsistent parameters but override_existing is false."
                    )
                else:
                    logger.info(
                        f"found existing {cls.__name__} file with inconsistent parameter  and override_existing=True. Delete old file and build new one."
                    )
                    os.remove(hdf_path)
                    obj._build(node_pos=node_pos)
        else:
            logger.info("no existing hdf file found. Build hdf...")
            obj._build(node_pos=node_pos)

        return obj

    def _build(self, node_pos: NodePositionWithRsdHdf):
        for app in self.apps:
            max_bw = app.get_max_application_bandwidth_in_bps()
            if max_bw is None:
                raise ValueError(
                    f"NodeTxData is configured to include maxApplicationBandwidth data \
                                 into hdf attributes but application {app} does not provide a \
                                 maxApplicationBandwidth"
                )
            # packets sent
            df_bytes = app.sim.sql.vector_ids_to_host(
                module_name=app.module_f(),
                vector_name="packetSent:vector(packetBytes)",
                name_columns=["hostId"],
                drop=["vectorId"],
                pull_data=True,
                value_name="tx_bytes",
            )
            # number of packets per burst
            df_burst = (
                df_bytes.groupby(["hostId", "time"])["tx_bytes"]
                .agg(["count", "sum"])
                .set_axis(["burst_num", "burst_size"], axis=1)
            )
            if node_pos is not None:
                df_bytes = node_pos.merge_rsd_id_on_host_time_interval(df_bytes)
                df_burst = node_pos.merge_rsd_id_on_host_time_interval(df_burst)

            # write to hdf
            self.hdf.write_frame(
                group=app.group_by_app("tx_bytes"),
                frame=df_bytes,
            )

            self.hdf.write_frame(
                group=app.group_by_app("tx_burst"),
                frame=df_burst,
            )
            rsd = node_pos is not None

            attr_iter = product(
                ["tx_bytes", "tx_burst"],
                [(self.ATTR_max_bw, max_bw), (self.ATTR_rsd, rsd)],
            )
            for g, (attr_k, attr_v) in attr_iter:
                self.hdf.set_attribute(attr_k, attr_v, group=app.group_by_app(g))

            # not needed anymore, reduce RAM footprint
            del df_bytes
            df_bytes = None
            del df_burst
            df_burst = None

            if app.sim.sql.vector_exists(
                module_name=app.module_f(path="scheduler"),
                vector_name="txInterval:vector",
            ):
                # application has interval scheduler, load interval data
                df_interval = app.sim.sql.vec_data_pivot(
                    module_name=app.module_f(path="scheduler"),
                    vector_name_map={
                        "txInterval:vector": {"name": "tx_interval", "dtype": float},
                        "txDetInterval:vector": {
                            "name": "tx_interval_det",
                            "dtype": float,
                        },
                        "txMemberValue:vector": {"name": "member_count", "dtype": int},
                    },
                ).droplevel("eventNumber")
                if node_pos is not None:
                    df_interval = node_pos.merge_rsd_id_on_host_time_interval(
                        df_interval
                    )

                self.hdf.write_frame(
                    group=app.group_by_app("tx_interval"), frame=df_interval
                )
                self.hdf.set_attribute(
                    self.ATTR_rsd, rsd, group=app.group_by_app("tx_interval")
                )
                self.hdf.set_attribute(
                    self.ATTR_max_bw, max_bw, group=app.group_by_app("tx_interval")
                )
                # not needed anymore, reduce RAM footprint
                del df_interval
                df_interval = None
        self.hdf.repack_hdf(keep_old_file=False)


class SimGroupNodeTxData:
    pass
