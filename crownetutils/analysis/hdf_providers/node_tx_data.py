from __future__ import annotations

import os
from typing import List

import pandas as pd

from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.analysis.hdf_providers.node_position import NodePositionWithRsdHdf
from crownetutils.analysis.hdf_providers.sql_app_proxy import SqlAppProxy
from crownetutils.analysis.omnetpp import OppAnalysis
from crownetutils.utils.logging import logger
from crownetutils.utils.misc import Timer


class NodeTxData:
    base_groups = ["tx_bytes", "tx_burst", "tx_interval"]

    def __init__(
        self,
        hdf_path: str,
        apps: List[SqlAppProxy],
    ) -> None:
        self.apps: List[SqlAppProxy] = apps
        self.hdf_path = hdf_path
        self._hdf: BaseHdfProvider = None

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
        return BaseHdfProvider(self.hdf_path, group=self.g(app, "tx_bytes"))

    def tx_burst(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        return BaseHdfProvider(self.hdf_path, group=self.g(app, "tx_burst"))

    def tx_interval(self, app: str | SqlAppProxy) -> BaseHdfProvider:
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

    def tx_throuput_diff_by_app(
        self,
        target_rates: dict,
        bin_size: float = 10.0,
        throughput_unit: float = 1000.0,
        serving_enb: int | None = None,
    ):
        if serving_enb is not None:
            serving_enb = f"servingEnb={serving_enb}"
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
        if obj.hdf.hdf_file_exists and obj._check_if_hdf_consitent():
            logger.info(
                "found existing hdf file with matching paramter setup. No build requiered."
            )
            return obj
        else:
            if obj.hdf.hdf_file_exists:
                if not override_existing:
                    raise ValueError(
                        "found existing hdf file with inconsitent paramters but override_existing is false."
                    )
                else:
                    logger.info(
                        "found existing hdf file with inconsitent paramter  and override_existing=True. Delete old file and build new one."
                    )
                    os.remove(hdf_path)
            else:
                logger.info("no hdf file found. Build hdf...")
            with Timer():
                obj._build(node_pos=node_pos)
            return obj

    def _check_if_hdf_consitent(self):
        if not self.hdf.hdf_file_exists:
            return False
        for app in self.apps:
            for g in self.base_groups:
                if not self.hdf.contains_group(app.group_by_app(g)):
                    return False
                if not self.hdf.get_attribute(
                    attr_key="with_rsd", group=app.group_by_app(g), default=False
                ):
                    return False
        return True

    def _build(self, node_pos: NodePositionWithRsdHdf):
        for app in self.apps:
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
            self.hdf.set_attribute("with_rsd", rsd, group=app.group_by_app("tx_bytes"))
            self.hdf.set_attribute("with_rsd", rsd, group=app.group_by_app("tx_burst"))

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
                    "with_rsd", rsd, group=app.group_by_app("tx_interval")
                )
                # not needed anymore, reduce RAM footprint
                del df_interval
                df_interval = None
        self.hdf.repack_hdf(keep_old_file=False)
