import logging
import os

logger = logging.getLogger(__name__)


class _DockerCfg:
    def __init__(self) -> None:
        self.default_docker_registry_public = "ghcr.io/rover-hm"
        self.default_docker_registry_dev = "sam-dev.cs.hm.edu:5023/rover/crownet"
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

    def full_path(self, container_name):
        return f"{self.registry}/{container_name}"

    def __call__(self, container_name) -> str:
        self.full_path(container_name)


DockerCfg = _DockerCfg()
