import os

from roveranalyzer.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)
from roveranalyzer.entrypoint.parser import ArgList
from roveranalyzer.simulators.opp.configuration import CrowNetConfig
from roveranalyzer.utils import logger, sockcheck


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
        use_local=False,
        scenario=None,
        result_dir="results",
        experiment_label="vadere_controlled_out",
        ctrl_args: ArgList = ArgList(),
    ):

        # if connection_mode == "client":
        #     if scenario is None:
        #         raise ("Scenario file must be provided in client mode.")

        # use python source code of flowcontroler instead of installed flowcontroler package in the container.
        # the /init_dev.sh script is part of the container an will install
        if use_local:
            exec_cmd = "/init_dev.sh"
            env = self.run_args.get("environment", {})
            env.setdefault("CROWNET_HOME", CrowNetConfig.path_crownet_home())
            self.run_args["environment"] = env
        else:
            exec_cmd = "python3"

        cmd = [
            exec_cmd,
            control_file,
            "--port",
            str(traci_port),
            "--host-name",
            host_name,
            ctrl_args.to_string(),
        ]

        if connection_mode == "client":
            cmd.extend(["--client-mode", "--scenario-file", scenario])
            cmd.extend(["--output-dir", result_dir])
            cmd.extend(["--experiment-label", experiment_label])

        logger.debug(f"start controller container(start_controller)")
        logger.debug(f"cmd: {' '.join(cmd)}")
        run_result = self.run(cmd, self.run_args)
        sockcheck.check(self.name, int(traci_port))
        return run_result
