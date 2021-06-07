import os
import pprint
from enum import Enum
from pathlib import Path
from typing import Union

import docker
from docker.errors import NotFound
from docker.models.containers import Container
from docker.types import LogConfig
from requests import ReadTimeout

from roveranalyzer.utils import logger


class ContainerLogWriter:
    def __init__(self, path):
        self.path = path

    def write(self, container_name, output: bytes, *args, **kwargs):
        logger.info(f"write output of {container_name}  to {self.path}")
        os.makedirs(os.path.split(self.path)[0], exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as log:
            log.write(output.decode("utf-8"))


class DockerCleanup(Enum):
    REMOVE = "remove"
    KEEP_FAILED = "keep_failed"
    KEEP = "keep"

    def __str__(self):
        return self.value


class DockerReuse(Enum):
    REUSE_STOPPED = "reuse_stopped"
    REUSE_RUNNING = "reuse_stopped"
    REMOVE_STOPPED = "remove_stopped"
    REMOVE_RUNNING = "remove_running"
    NEW_ONLY = "new_only"

    def __str__(self):
        return self.value


def stop_containers(names):
    c = docker.DockerClient = docker.from_env()
    for container in names:
        try:
            c = c.containers.get(container)
            c.stop(timeout=2)
        except Exception:
            pass


class DockerRunner:
    NET = "rovernet"

    def __init__(
        self,
        image,
        tag="latest",
        docker_client=None,
        name="",
        cleanup_policy=DockerCleanup.KEEP_FAILED,
        reuse_policy=DockerReuse.NEW_ONLY,
        detach=False,
        journal_tag="",
    ):
        if docker_client is None:
            self.client: docker.DockerClient = docker.from_env()
        else:
            self.client: docker.DockerClient = docker_client
        self.image = f"{image}:{tag}"
        self.rep = image
        self.tag = tag
        self.user_home = str(Path.home())
        self.user_id = os.getuid()
        self.working_dir = str(Path.cwd())
        self.name = name
        self.journal_tag = journal_tag
        self.hostname = name
        self.detach = detach
        self.cleanupPolicy = cleanup_policy
        self.reuse_policy = reuse_policy
        self.run_args = {}
        self._container = None
        self._log_clb: Union[
            ContainerLogWriter, None
        ] = None  # callback to handle logoutput of container

        # last call in init.
        self._apply_default_volumes()
        self._apply_default_environment()
        self.set_run_args()
        if self.journal_tag != "":
            self.set_log_driver(
                LogConfig(
                    type=LogConfig.types.JOURNALD, config={"tag": self.journal_tag}
                )
            )

    @property
    def container(self):
        return self._container

    @container.setter
    def container(self, val):
        self._container = val

    def apply_reuse_policy(self):
        try:
            if self.name == "":
                return
            _container = self.client.containers.get(self.name)
            reuse_policy = self.reuse_policy

            if reuse_policy == DockerReuse.REMOVE_RUNNING:
                _container.stop()
                _container.remove()
                logger.info(
                    f"stop and remove existing container with name '{self.name}'"
                )
            elif reuse_policy == DockerReuse.REMOVE_STOPPED:
                if _container.status == "running":
                    raise ValueError(
                        f"container is still running but reuse policy is {reuse_policy}."
                    )
                _container.remove()
                logger.info(f"remove existing container with name '{self.name}'")
            elif (
                reuse_policy == DockerReuse.REUSE_STOPPED
                or reuse_policy == DockerReuse.REUSE_RUNNING
            ):
                if _container.status == "running":
                    _container.stop()
                    logger.info(f"stop existing container with name '{self.name}'")
            elif reuse_policy == DockerReuse.NEW_ONLY:
                # container exists. --> error here
                raise ValueError(
                    f"container exists with status: {_container.status}. Reuse policy {reuse_policy} "
                    f"requires that container does not exist."
                )
            else:
                raise ValueError(f"unknown reuse policy provided {reuse_policy}")

        except NotFound as notFoundErr:
            pass  # ignore do nothing

    def _apply_default_volumes(self):
        self.volumes: dict = {
            self.user_home: {"bind": self.user_home, "mode": "rw"},
            "/etc/group": {"bind": "/etc/group", "mode": "ro"},
            "/etc/passwd": {"bind": "/etc/passwd", "mode": "ro"},
            "/etc/shadow": {"bind": "/etc/shadow", "mode": "ro"},
            "/etc/sudoers.d": {"bind": "/etc/sudoers.d", "mode": "ro"},
        }
        # bin X11 in container if DISPLAY is available
        if "DISPLAY" in os.environ:
            self.volumes["/tmp/.X11-unix"] = {
                "bind": "/tmp/.X11-unix",
                "mode": "rw",
            }

        if "OPP_EXTERN_DATA_MNT" in os.environ:
            _mnt = os.environ["OPP_EXTERN_DATA_MNT"].split(":")
            if len(_mnt) != 3:
                raise ValueError(
                    f"expected 3 tuple seperated by >:< link /mnt/foo:/mnt/bar:rw but"
                    f" got {os.environ['OPP_EXTERN_DATA_MNT']}"
                )
            self.volumes[_mnt[0]] = {"bind": _mnt[1], "mode": _mnt[2]}

    def set_runargs(self, key, value):
        self.run_args[key] = value

    def set_working_dir(self, wdir):
        self.run_args["working_dir"] = wdir

    def set_log_callback(self, clb: ContainerLogWriter):
        self._log_clb = clb

    def _apply_default_environment(self):
        self.environment: dict = {}
        if "DISPLAY" in os.environ:
            self.environment["DISPLAY"] = os.environ.get("DISPLAY")

    def check_create_network(self):
        existing_networks = [n.name for n in self.client.networks.list()]
        if self.NET not in existing_networks:
            self.client.networks.create(self.NET)
        return self.NET

    def set_run_args(self, run_args=None):
        """
        override default run args. If run_args is None use defaults.
        """
        if run_args is None:
            self.run_args: dict = {
                "cap_add": ["SYS_PTRACE"],
                "user": self.user_id,
                "network": self.check_create_network(),
                "environment": self.environment,
                "volumes": self.volumes,
                "working_dir": self.working_dir,
            }
        else:
            self.run_args = run_args

    def set_log_driver(self, driver: LogConfig):
        self.run_args["log_config"] = driver

    def build_run_args(self, **run_args):
        """
        add dynamic setup and override defaults with run_args if any given.
        """
        # if name or host not set let docker choose otherwise set given values
        if self.name != "":
            self.run_args["name"] = self.name
        if self.hostname != "":
            self.run_args["hostname"] = self.hostname
        self.run_args.update(run_args)

    def wrap_command(self, cmd):
        if type(cmd) == list:
            return f'/bin/bash -c "cd {self.working_dir}; {" ".join(cmd)}"'
        else:
            return f'/bin/bash -c "cd {self.working_dir}; {cmd}"'

    def pull_images(self):

        try:
            self.client.images.get(self.image)
        except:
            logger.info(
                f"Docker image is missing. Try to pull {self.image} from repository."
            )
            self.client.images.pull(repository=self.rep, tag=self.tag)

    def parse_log(self):
        if self._log_clb is not None:
            if self._container is None:
                logger.warning(
                    f"no container found for dockerrunner. Did you call parse_log at the wrong time?"
                )
            else:
                self._log_clb.write(self.name, self._container.logs())

    def create_container(self, cmd="/init.sh", **run_args) -> Container:
        """
        run container. If no command is given execute the default entry point '/init.sh'
        """
        self.build_run_args(**run_args)  # set name if given
        command = self.wrap_command(cmd)
        logger.info(f"{'#'*10} create container [image:{self.image}]")
        logger.debug(f"cmd: \n{pprint.pformat(command, indent=2)}")
        logger.debug(f"runargs: \n{pprint.pformat(self.run_args, indent=2)}")

        c: Container = self.client.containers.create(
            image=self.image, command=command, **self.run_args
        )
        logger.info(f"{'#'*10} container created {c.name} [image:{self.image}]")
        return c

    def wait(self, timeout=-1):
        if timeout > 0:
            ret = self._container.wait(timeout=timeout)
        else:
            ret = self._container.wait()
        if ret["StatusCode"] != 0:
            logger.error(f"{'#' * 80}")
            logger.error(f"{'#' * 80}")
            logger.error(f"Command returned {ret['StatusCode']}")
            if (
                "log_config" in self.run_args
                and self.run_args["log_config"].type == LogConfig.types.JOURNALD
            ):
                logger.error(
                    f"For full container output see: journalctl -b CONTAINER_TAG={self.journal_tag} --all"
                )
            container_log = self._container.logs()
            if container_log != b"":
                logger.error(f'Container Log:\n {container_log.decode("utf-8")}')
            logger.error(f"{'#' * 80}")
            logger.error(f"{'#' * 80}")
        return ret

    def run(self, cmd, perform_cleanup=True, **run_args):
        err = False

        self.pull_images()

        try:
            self._container = self.create_container(cmd, **run_args)
            logger.info(
                f"start container {self._container.name} with {{detach: {self.detach},"
                f" remove: {self.cleanupPolicy}, journal_tag: {self.journal_tag}}}"
            )
            self._container.start()
            if not self.detach:
                ret = self.wait()
            else:
                ret = {"Error": None, "StatusCode": 0}
        except (KeyboardInterrupt, SystemExit) as kInter:
            logger.warning(
                f"KeyboardInterrupt. Stop Container {self._container.name} ..."
            )
            err = True  # stop but do not delete container
            raise  # re-raise so other parts can react to SIGINT
        except ReadTimeout as e:
            logger.error("wait tdimeout reached")
            err = True  # stop but do not delete container
            raise RuntimeError(e)
        except docker.errors.APIError as e:
            logger.error(f"some API error occurred")
            logger.error(e.explanation)
            err = True  # stop but do not delete container
            raise RuntimeError(e)
        except BaseException as e:
            logger.error(f"some error occurred")
            err = True  # stop but do not delete container
            raise RuntimeError(e)
        finally:
            if not self.detach and perform_cleanup:
                self.container_cleanup(has_error_state=err)
        return ret

    def container_cleanup(self, has_error_state=False):
        """
        stop and remove based on cleanupPolicy.
        """
        if self._container is None:
            return  # do nothing

        self.parse_log()
        logger.debug(f"Stop container {self._container.name} ...")
        self._container.stop()
        if self.cleanupPolicy == DockerCleanup.REMOVE or (
            self.cleanupPolicy == DockerCleanup.KEEP_FAILED and not has_error_state
        ):
            logger.debug(f"remove container {self._container.name} ...")
            self._container.remove()
            self._container = None  # container removed so do not keep reference

    def container_stop_and_remove(self):
        """
        stop and remove and ignore cleanupPolicy.
        """
        if self._container is not None:
            self._container.stop()
            self._container.remove()
            self._container = None

    def __str__(self) -> str:
        return f"{__name__}: Run Arguments: {pprint.pformat(self.run_args, indent=2)}"
