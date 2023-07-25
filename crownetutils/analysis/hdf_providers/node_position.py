from __future__ import annotations

from functools import partial

from crownetutils.analysis.common import Simulation
from crownetutils.analysis.hdf.provider import BaseHdfProvider
from crownetutils.omnetpp.scave import CrownetSql


class NodePositionHdf(BaseHdfProvider):
    @classmethod
    def get(cls, hdf_path: str, sql: CrownetSql, **kwargs) -> NodePositionHdf:
        obj = cls(hdf_path=hdf_path, group="trajectories", allow_lazy_loading=True)
        if len(kwargs) > 0:
            obj.add_group_factory(obj.group, partial(sql.node_position, **kwargs))
        else:
            obj.add_group_factory(obj.group, sql.node_position)
        return obj

    @classmethod
    def from_sim(cls, sim: Simulation, **kwargs) -> NodePositionHdf:
        return cls.get(sim.path("position.h5"), sim.sql, **kwargs)


class EnbPositionHdf(BaseHdfProvider):
    @classmethod
    def get(cls, hdf_path: str, sql: CrownetSql, **kwargs) -> EnbPositionHdf:
        obj = cls(hdf_path=hdf_path, group="enb", allow_lazy_loading=True)
        if len(kwargs) > 0:
            obj.add_group_factory(obj.group, partial(sql.enb_position, **kwargs))
        else:
            obj.add_group_factory(obj.group, sql.enb_position)
        return obj

    @classmethod
    def from_sim(cls, sim: Simulation, **kwargs) -> EnbPositionHdf:
        return cls.get(sim.path("position.h5"), sim.sql, **kwargs)
