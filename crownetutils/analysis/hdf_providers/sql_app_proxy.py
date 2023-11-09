from dataclasses import dataclass

from crownetutils.analysis.common import Simulation
from crownetutils.omnetpp.sql import RelativeModuleFunction


@dataclass
class SqlAppProxy:
    name: str
    module_f: RelativeModuleFunction
    sim: Simulation

    def group_by_app(self, path):
        return f"{self.name}/{path}"
