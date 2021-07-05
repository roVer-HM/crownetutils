import os

from roveranalyzer.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)
from roveranalyzer.utils import logger


class SumoRunner(DockerRunner):
    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/crownet/sumo",
        tag="latest",
        docker_client=None,
        name="",
        cleanup_policy=DockerCleanup.REMOVE,
        reuse_policy=DockerReuse.REMOVE_STOPPED,
        detach=False,
        journal_tag="",
    ):
        super().__init__(
            image,
            tag,
            docker_client=docker_client,
            name=name,
            cleanup_policy=cleanup_policy,
            reuse_policy=reuse_policy,
            detach=detach,
            journal_tag=journal_tag,
        )

    def _apply_default_volumes(self):
        super()._apply_default_volumes()
        # add...

    def _apply_default_environment(self):
        super()._apply_default_environment()
        # add...

    def set_run_args(self, run_args=None):
        super().set_run_args()
        # add...

    def exec_single_server(
        self,
        config_path,
        traci_port=9999,
        message_log=os.devnull,
        run_args_override=None,
    ):
        cmd = [
            "sumo",
            "-v",
            "--remote-port",
            str(traci_port),
            "--configuration-file",
            config_path,
            "--message-log",
            message_log,
            "--no-step-log",
            "--quit-on-end",
        ]

        if run_args_override is None:
            run_args_override = {}

        logger.debug(f"start sumo container(single server)")
        logger.debug(f"cmd: {' '.join(cmd)}")
        return self.run(cmd, **run_args_override)

    def single_launcher(
        self,
        traci_port=9999,
        message_log=os.devnull,
        run_args_override=None,
    ):
        cmd = [
            "/veins_launchd",
            "-vvv",
            "--port",
            str(traci_port),
            "--bind",
            "0.0.0.0",
            "--logfile",
            message_log,
            "--single-run",
        ]
        if run_args_override is None:
            run_args_override = {}

        logger.debug(f"start sumo container(single server)")
        logger.debug(f"cmd: {' '.join(cmd)}")
        return self.run(cmd, **run_args_override)

    def exec_start_vadere_laucher(self):
        """
        start the vadere-laucher.py script in the container which creates multiple Vadere
        instances inside ONE container.
        """
        pass

    def exec_vadere_gui(self):
        """
        start vadere gui to create or execute vadere scenarios.
        """
        pass
