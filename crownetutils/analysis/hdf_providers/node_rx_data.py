from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.analysis.hdf_providers.node_position import NodePositionWithRsdHdf
from crownetutils.analysis.hdf_providers.sql_app_proxy import SqlAppProxy
from crownetutils.utils.logging import logger, timing
from crownetutils.utils.misc import Timer


class NodeRxData:
    G_BY_SRC_STATS = "rcvd_stats"
    G_BY_APP_STATS = "rcvd_by_app"
    base_groups = [G_BY_SRC_STATS, G_BY_APP_STATS]

    def __init__(
        self,
        hdf_path: str,
        apps: List[SqlAppProxy],
        node_pos: NodePositionWithRsdHdf = None,
    ) -> None:
        self.hdf_path = hdf_path
        self.apps: List[SqlAppProxy] = apps
        self.node_pos = node_pos
        self._hdf: BaseHdfProvider | None = None

    @classmethod
    def get_or_create(
        cls,
        hdf_path: str,
        apps: List[SqlAppProxy],
        node_pos: NodePositionWithRsdHdf = None,
        override_existing: bool = True,
    ) -> NodeRxData:
        obj: NodeRxData = cls(hdf_path, apps, node_pos)
        if obj.hdf.hdf_file_exists and obj._check_hdf_consitency():
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
                obj._build()
            return obj

    @property
    def hdf(self) -> BaseHdfProvider:
        if self._hdf is None:
            self._hdf = BaseHdfProvider(self.hdf_path)
        return self._hdf

    def rcvd_data(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        return BaseHdfProvider(self.hdf_path, group=self.g(app, self.G_BY_SRC_STATS))

    def rcvd_by_app(self, app: str | SqlAppProxy) -> BaseHdfProvider:
        return BaseHdfProvider(self.hdf_path, group=self.g(app, self.G_BY_APP_STATS))

    def g(self, app: str | SqlAppProxy, path: str) -> str:
        if isinstance(app, SqlAppProxy):
            return app.group_by_app(path)
        else:
            return f"{app}/{path}"

    def all_apps(
        self,
        where: str | None = None,
        columns: List[str] | None = None,
        with_app_names: bool = True,
    ) -> pd.DataFrame:
        ret = []
        for app in self.apps:
            g = app.group_by_app(self.G_BY_SRC_STATS)
            df = self.hdf.select(key=g, where=where, columns=columns)
            app_ids = self.hdf.get_attribute("app_ids", group=g, default=None)
            if with_app_names:
                if app_ids is None:
                    raise ValueError("No app names found in hdf attributes")
                else:
                    df["app"] = app.name
            ret.append(df)

        ret = pd.concat(ret, axis=0, verify_integrity=False)
        return ret

    def _check_hdf_consitency(self) -> bool:
        if not self.hdf.hdf_file_exists:
            return False
        for app in self.apps:
            for g in self.base_groups:
                if not self.hdf.contains_group(app.group_by_app(g)):
                    return False
                # if not self.hdf.has_attribute("app_ids", group=app.group_by_app(g)):
                #     return False
        return True

    def _build(self):
        app_names = [app.name for app in self.apps]
        app_names.sort()
        app_ids = {n: i for i, n in enumerate(app_names)}

        for app in self.apps:
            # serach for used vecor name version
            seqNo_vec = ["rcvdPkSeqNo:vector", "rcvdPktPerSrcSeqNo:vector"]
            seqNo_vec = app.sim.sql.find_vector_name(app.module_f(), seqNo_vec)
            vec_names = {
                seqNo_vec: {
                    "name": "seqNo",
                    "dtype": np.int32,
                },
                "rcvdPktPerSrcLossCount:vector": dict(
                    name="pkt_loss_sum", dtype=np.int32
                ),
                "rcvdPktPerSrcCount:vector": dict(
                    name="total_pkt_received", dtype=np.int32
                ),
                "rcvdPkHostId:vector": dict(name="srcHostId", dtype=np.int32),
                "rcvdPktPerSrcJitter:vector": dict(name="jitter", dtype=np.float32),
                "rcvdPkLifetime:vector": dict(name="delay", dtype=np.float32),
                "packetReceived:vector(packetBytes)": dict(
                    name="pkt_bytes", dtype=np.float32
                ),
            }
            vec_data = app.sim.sql.vec_data_pivot(
                module_name=app.module_f(),
                vector_name_map=vec_names,
                append_index=["srcHostId"],
            ).sort_index()

            # drop self messages, where hostId == srcHostId
            _shape = vec_data.shape
            _self_msg_mask = vec_data.index.get_level_values(
                "hostId"
            ) != vec_data.index.get_level_values("srcHostId")
            vec_data = vec_data[_self_msg_mask].copy(deep=True)
            logger.info(
                f"remove self references (hostId==srcHostId): {_shape}->{vec_data.shape} "
            )

            # packet loss
            vec_data["total_pkt_send"] = (
                vec_data["pkt_loss_sum"] + vec_data["total_pkt_received"]
            )
            vec_data["PRR"] = (
                vec_data["total_pkt_received"] / vec_data["total_pkt_send"]
            )
            vec_data["app"] = app_ids[app.name]

            if self.node_pos is not None:
                vec_data = self.node_pos.merge_rsd_id_on_host_time_interval(vec_data)

            self.hdf.write_frame(
                group=app.group_by_app(self.G_BY_SRC_STATS),
                frame=vec_data,
            )
            self.hdf.set_attribute(
                attr_key="app_ids",
                value=app_ids,
                group=app.group_by_app(self.G_BY_SRC_STATS),
            )

            # by app (data used in interval calculation)
            vec_names = {
                "rcvdPktAvgSize:vector": dict(name="avg_pkt_size", dtype=np.float32),
                "rcvdPktCount:vector": dict(name="pkt_count", dtype=np.float32),
            }
            vec_data = (
                app.sim.sql.vec_data_pivot(
                    module_name=app.module_f(),
                    vector_name_map=vec_names,
                )
                .reset_index("eventNumber")
                .sort_index()
            )
            if self.node_pos is not None:
                vec_data = self.node_pos.merge_rsd_id_on_host_time_interval(vec_data)

            self.hdf.write_frame(
                group=app.group_by_app(self.G_BY_APP_STATS),
                frame=vec_data,
            )
            self.hdf.set_attribute(
                attr_key="app_ids",
                value=app_ids,
                group=app.group_by_app(self.G_BY_APP_STATS),
            )

        self.hdf.repack_hdf(keep_old_file=False)
