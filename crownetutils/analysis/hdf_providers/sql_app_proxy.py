from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

from crownetutils.analysis.common import Simulation
from crownetutils.omnetpp.sql import RelativeModuleFunction
from crownetutils.utils.units import str_to_bps


class _Empty:
    pass


__empty_default = _Empty()


@dataclass
class SqlAppProxy:
    name: str
    module_f: RelativeModuleFunction
    sim: Simulation
    f_dict: dict = field(default_factory=dict, init=False)

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

    def add_callback(self, cb_name, func, *args, **kwargs) -> SqlAppProxy:
        """Add some callback with or without partial arguments be saved in the SqlAppProxy object."""
        if len(args) > 0 or len(kwargs) > 0:
            self.f_dict[cb_name] = partial(func, *args, **kwargs)
        else:
            self.f_dict[cb_name] = func
        return self

    def call_cb(self, cb_name, default=_Empty()):
        """Call callback or provide some default value. Note 'None' is a valid default value.
        If no default value is provided and the cb_name does not exist an exception is thrown
        """
        if cb_name not in self.f_dict:
            if isinstance(default, _Empty):
                raise ValueError(
                    "callback name not found and no default value provided"
                )
            else:
                return None
        else:
            return self.f_dict[cb_name]()
