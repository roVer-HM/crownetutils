from roveranalyzer.analysis.common import Simulation
from roveranalyzer.analysis.hdf.provider import BaseHdfProvider
from roveranalyzer.omnetpp.scave import CrownetSql


class NodePositionHdf(BaseHdfProvider):
    @classmethod
    def get(cls, hdf_path: str, sql: CrownetSql):
        obj = cls(hdf_path=hdf_path, group="trajectories", allow_lazy_loading=True)
        obj.add_group_factory(obj.group, sql.node_position)
        return obj

    @classmethod
    def from_sim(cls, sim: Simulation):
        return cls.get(sim.path("position.h5"), sim.sql)
