import argparse
import os

from crownetutils.dockerrunner import DockerCfg
from crownetutils.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)
from crownetutils.utils import sockcheck
from crownetutils.utils.logging import logger


class VadereRunner(DockerRunner):
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
        image=DockerCfg("vadere"),
        tag=DockerCfg.get_default_tag(DockerCfg.VAR_VADERE_TAG),
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
        traci_port=9998,
        loglevel=LogLevel.DEBUG,
        logfile=os.devnull,
        show_gui=False,
        run_args_override=None,
        output_dir=None,
    ):
        """
        start Vadere server waiting for exactly ONE connection on given traci_port. After
        simulation returns the container will stop.
        """

        cmd = [
            "java",
            "-jar",
            "/opt/vadere/vadere/VadereManager/target/vadere-server.jar",
            "--loglevel",
            loglevel,
            "--logname",
            logfile,
            "--port",
            str(traci_port),
            "--bind",
            "0.0.0.0",
            "--single-client",
        ]
        if show_gui:
            cmd.append("--gui-mode")

        if output_dir != None:
            cmd.extend(["--output-dir", output_dir])

        if run_args_override is None:
            run_args_override = {}

        logger.debug(f"start vadere container(single server)")
        logger.debug(f"cmd: {' '.join(cmd)}")
        run_result = self.run(cmd, **run_args_override)
        self.wait_for_log(f"listening on port {traci_port}...")
        return run_result

    def exec_vadere_only(self, scenario_file, output_path, run_args_override=None):

        cmd = [
            "java",
            "-jar",
            "/opt/vadere/vadere/VadereSimulator/target/vadere-console.jar",
            "suq",
            "-f",
            scenario_file,
            "-o",
            output_path,
        ]

        if run_args_override is None:
            run_args_override = {}

        logger.debug(f"start vadere container(exec_vadere_only")
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


def add_vadere_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-sf",
        "--scenario-file",
        dest="scenario_file",
        default="",
        required=False,
        help="Scenario-file *.scenario for Vadere simulation.",
    )
    parser.add_argument(
        "--create-vadere-container",
        dest="create_vadere_container",
        action="store_true",
        default=False,
        required=False,
        help="If set a vadere container with name vadere_<run-name> is created matching to opp_<run-name> container.",
    )
    parser.add_argument(
        "--v.wait-timeout",
        dest="v_wait_timeout",
        default=360,
        required=False,
        help="Time to wait for vadere container to close after OMNeT++ container has finished. Default=360s",
    )
    parser.add_argument(
        "--v.traci-port",
        dest="v_traci_port",
        default="9998",
        required=False,
        help="Set TraCI Port in Vadere container. (Default: 9998)",
    )
    parser.add_argument(
        "--vadere-tag",
        dest="vadere_tag",
        default=DockerCfg.get_default_tag(DockerCfg.VAR_VADERE_TAG),
        required=False,
        help=f"Choose Vadere container. (Default: {DockerCfg.get_default_tag(DockerCfg.VAR_VADERE_TAG)})",
    )
    parser.add_argument(
        "--v.loglevel",
        dest="v_loglevel",
        default="INFO",
        required=False,
        help="Set loglevel of (Vadere)TraCI Server [WARN, INFO, DEBUG, TRACE]. (Default: INFO)",
    )
    parser.add_argument(
        "--v.logfile",
        dest="v_logfile",
        default="",
        required=False,
        help="Set log file name of Vadere. If not set '', log file will not be created. "
        "This setting has no effect on --log-journald. (Default: '') ",
    )
