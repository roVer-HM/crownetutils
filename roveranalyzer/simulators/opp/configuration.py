import os
import pathlib
from pathlib import Path

from roveranalyzer.utils.path import PathHelper

_default_scavetool_cmd = "opp_scavetool"
_default_use_docker_container = True

if "CROWNET_HOME" in os.environ:
    _default_crownet = os.environ.get("CROWNET_HOME")
    _default_opp_container_path = os.path.join(
        os.environ.get("CROWNET_HOME"), "scripts/omnetpp"
    )
else:

    _default_crownet = os.path.join(str(Path.home().absolute()), "crownet")
    if not os.path.exists(_default_crownet):
        print(
            f"CROWNET_HOME not set and not in default location. Deactivate docker support"
        )
        _default_use_docker_container = False
        _default_opp_container_path = ""
    else:
        print(f"CROWNET_HOME not set, use  default {_default_crownet}")
        _default_opp_container_path = os.path.join(_default_crownet, "scripts/omnetpp")


class ConfigException(Exception):
    pass


def check_setup(_cls):
    if "CROWNET_HOME" not in os.environ:
        raise ValueError("CROWNET_HOME not set.")
    return _cls


class ScaveConfig:
    """
    Config object to determine the correct way to call the scavetool.
    """

    def __init__(self, **kwargs):

        self.scave_tool_cmd = kwargs.get("scave_tool_cmd", _default_scavetool_cmd)
        self.use_docker_container = kwargs.get(
            "use_docker_container", _default_use_docker_container
        )
        self.opp_container_path = kwargs.get(
            "opp_container_path", _default_opp_container_path
        )
        self.rover_main = kwargs.get("crownet", _default_crownet)

    def scave_cmd(self, silent=False):
        if self.use_docker_container:
            if silent:
                return [self.opp_container_path, "execs", self.scave_tool_cmd]
            else:
                return [self.opp_container_path, "exec", self.scave_tool_cmd]
        else:
            return [self.scave_tool_cmd]

    @property
    def uses_container(self):
        return self.use_docker_container

    def escape(self, val):
        """Escape characters if docker ist used."""
        if self.uses_container:
            val = val.replace(r" ", r"\ ")
            val = val.replace(r"(", r"\(")
            val = val.replace(r")", r"\)")
        print(f"escaped: {val}")
        return val
