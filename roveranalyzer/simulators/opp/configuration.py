import os
import pathlib
from pathlib import Path

_default_scavetool_cmd = "scavetool"
_default_use_docker_container = True

if "CROWNET_HOME" in os.environ:
    _default_rover_main = os.environ.get("CROWNET_HOME")
    _default_opp_container_path = os.path.join(
        os.environ.get("CROWNET_HOME"), "scripts/omnetpp"
    )
else:

    _default_rover_main = os.path.join(str(Path.home().absolute()), "rover-main")
    if not os.path.exists(_default_rover_main):
        print(
            f"CROWNET_HOME not set and not in default location. Deactivate docker support"
        )
        _default_use_docker_container = False
        _default_opp_container_path = ""
    else:
        print(f"CROWNET_HOME not set, use  default {_default_rover_main}")
        os.path.join(_default_rover_main, "scripts/omnetpp")


class ConfigException(Exception):
    pass


def check_setup(_cls):
    if "CROWNET_HOME" not in os.environ:
        raise ValueError("CROWNET_HOME not set.")
    return _cls


@check_setup
class RoverConfig:
    NAME_PACKAGE = "roveranalyzer"
    NAME_ROVER_CONFIG_FILE = "roveranalyzer.conf"

    @classmethod
    def path_userhome(cls):
        return pathlib.Path.home()

    @classmethod
    def path_rover_main(cls):
        return os.environ.get("CROWNET_HOME")

    @classmethod
    def join_rover_main(cls, other):
        return os.path.join(cls.path_rover_main(), other)


class Config:
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
        self.rover_main = kwargs.get("rover_main", _default_rover_main)

    @property
    def scave_cmd(self):
        if self.use_docker_container:
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
        return val
