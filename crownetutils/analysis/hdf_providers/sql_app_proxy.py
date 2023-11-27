from dataclasses import dataclass

from crownetutils.analysis.common import Simulation
from crownetutils.omnetpp.sql import RelativeModuleFunction
from crownetutils.utils.units import str_to_bps


@dataclass
class SqlAppProxy:
    name: str
    module_f: RelativeModuleFunction
    sim: Simulation

    def group_by_app(self, path):
        return f"{self.name}/{path}"

    def max_application_bandwidth(self) -> bool:
        return self.get_max_application_bandwidth_in_bps() is not None

    def get_max_application_bandwidth_in_bps(self, default=None) -> int:
        m = self.module_f(path="scheduler")
        maxAppBw = self.sim.sql.get_run_parameter(
            module_name=m, name="maxApplicationBandwidth", full_match=True
        )
        if maxAppBw.empty:
            return default
        value = maxAppBw["paramValue"].unique()
        if len(value) > 1:
            raise ValueError(
                f"found multiple maxApplicationBandwidth values for at path {m}. Got {value}"
            )

        return str_to_bps(value[0])
