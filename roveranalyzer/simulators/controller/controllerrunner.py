import logging
import os

from roveranalyzer.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)


class ControlRunner(DockerRunner):
    class LogLevel:
        OFF = "OFF"
        FATAL = "FATAL"
        ERROR = "ERROR"
        WARN = "WARN"
        INFO = "INFO"
        DEBUG = "DEBUG"
        TRACE = "TRACE"
        ALL = "ALL"

    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/crownet/flowcontrol",
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

    def start_controller(
        self,
        loglevel=LogLevel.DEBUG,
        logfile=os.devnull,
        run_args_override=None,
        control_file="control.py",
        host_name="vadere_rover_run",
        connection_mode="client",
        traci_port=9999,
        scenario=None,
    ):

        # if connection_mode == "client":
        #     if scenario is None:
        #         raise ("Scenario file must be provided in client mode.")

        cmd = [
            "python3",
            control_file,
            "--port",
            str(traci_port),
            "--host-name",
            host_name,
        ]

        if connection_mode == "client":
            cmd.extend(["--client-mode"])

        return self.run(cmd, self.run_args)
