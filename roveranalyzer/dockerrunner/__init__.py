import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class SimulationDispatcher(Protocol):
    """Dispatcher to run different combination of CrowNet supported simulators"""

    def run_vadere(self) -> int:
        """Run only a vadere simulation"""
        ...

    def run_simulation_vadere_ctl(self) -> int:
        """Run coupled (controlled) vadere simulation"""
        ...

    def run_simulation_vadere_omnet_ctl(self) -> int:
        """Run coupled simulation with vadere, control and omnet"""
        ...

    def run_simulation_omnet_vadere(self) -> int:
        """Run coupled simulation with omnet and vadere"""
        ...

    def run_simulation_omnet_sumo(self) -> int:
        """Run coupled simulation with omnet and sumo"""
        ...

    def run_simulation_omnet(self) -> int:
        """Run only a omnet simulation (without external mobility provider)"""
        ...

    def run_postprocessing_only(self) -> int:
        """Do not execute the simulations but execute post processing of each run."""
        ...


class _DockerCfg:

    VAR_OPP_TAG = "CROWNET_OPP_CONT_TAG"
    VAR_SUMO_TAG = "CROWNET_SUMO_CONT_TAG"
    VAR_VADERE_TAG = "CROWNET_VADERE_CONT_TAG"
    VAR_CONTROL_TAG = "CROWNET_CONTROL_CONT_TAG"

    def __init__(self) -> None:
        self.default_docker_registry_public = "ghcr.io/rover-hm"
        self.default_docker_registry_dev = "sam-dev.cs.hm.edu:5023/rover/crownet"
        self.default_docker_tag = "latest"
        if "CROWNET_IMAGE_BASE" in os.environ:
            self.registry = os.environ["CROWNET_IMAGE_BASE"]
            logger.debug(
                f"Set default container registry from environment: {self.registry}"
            )
        else:
            logger.debug(
                f"CROWNET_IMAGE_BASE variable not set default to public registry: '{self.default_docker_registry_public}'"
            )
            self.registry = self.default_docker_registry_public

    def get_default_tag(self, name):
        if name in os.environ:
            tag = os.environ[name]
            logger.debug(f"Set default container tag from environment: {tag}")
            return tag
        else:
            logger.debug(
                f"{name} variable not set default to: '{self.default_docker_tag}'"
            )
            return self.default_docker_tag

    def full_path(self, container_name):
        return f"{self.registry}/{container_name}"

    def __call__(self, container_name) -> str:
        return self.full_path(container_name)

    def print_settings(self):
        print(f"Registry: {self.registry}")


DockerCfg = _DockerCfg()
