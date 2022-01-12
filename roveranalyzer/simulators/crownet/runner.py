import argparse
import os
import signal
import sys
import time
from datetime import datetime
from typing import Any, List

import docker
from requests.exceptions import ReadTimeout

from roveranalyzer.dockerrunner.dockerrunner import (
    ContainerLogWriter,
    DockerCleanup,
    DockerClient,
    DockerReuse,
)
from roveranalyzer.entrypoint.parser import (
    ArgList,
    SimulationArgAction,
    SubstituteAction,
    filter_options,
)
from roveranalyzer.simulators.controller.controllerrunner import ControlRunner
from roveranalyzer.simulators.opp.configuration import CrowNetConfig
from roveranalyzer.simulators.opp.runner import OppRunner
from roveranalyzer.simulators.sumo.runner import SumoRunner
from roveranalyzer.simulators.vadere.runner import VadereRunner
from roveranalyzer.utils import levels, logger, set_format, set_level


def add_base_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--qoi", action="append", nargs="+", help="specify qoi files", type=str
    )
    parser.add_argument(
        "--pre",
        action="append",
        nargs="+",
        help="specify preprocessing methods",
        type=str,
    )
    parser.add_argument(
        "--resultdir",
        dest="result_dir",
        default="results",
        required=False,
        help="Base result directory used by all containers. Default: results",
    )
    parser.add_argument(
        "--write-container-log",
        dest="write_container_log",
        default=False,
        required=False,
        action="store_true",
        help="If true save output of containers in result dir <result>/container_<name>.out ",
    )
    parser.add_argument(
        "--experiment-label",
        dest="experiment_label",
        default="timestamp",
        action=SubstituteAction,
        do_on=["timestamp"],
        sub_action=lambda x: datetime.now()
        .isoformat()
        .replace("-", "")
        .replace(":", ""),
        required=False,
        help="experiment-label used in the result path. Use 'timestamp' to get current sanitized ISO-Format timestamp.",
    )

    parser.add_argument(
        "--override-host-config",
        dest="override-host-config",
        default=False,
        required=False,
        action="store_true",
        help="If set use --run-name as container names and override TraCI config parameters set in omnetpp.ini file.",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        nargs="?",
        default="rover_run",
        help="Set name of current run. This will be CONTAINER_TAG for journald. Default: rover_run",
    )
    parser.add_argument(
        "--cleanup-policy",
        dest="cleanup_policy",
        type=DockerCleanup,
        choices=list(DockerCleanup),
        default=DockerCleanup.REMOVE,
        required=False,
        help="select what to do with container that are done.",
    )
    parser.add_argument(
        "--reuse-policy",
        dest="reuse_policy",
        type=DockerCleanup,
        choices=list(DockerReuse),
        default=DockerReuse.REMOVE_RUNNING,
        required=False,
        help="select policy to reuse or remove existing running or stopped containers.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Set verbosity of command. From warnings and errors only (-v) to debug output (-vvv)",
    )


def add_control_arguments(parser: argparse.ArgumentParser, args: List[str]):
    parser.add_argument(
        "--control-tag",
        dest="control_tag",
        default="latest",
        required=False,
        help="Choose Control container. (Default: latest)",
    )
    parser.add_argument(
        "--ctrl.xxx",
        *filter_options(args, "--ctrl."),
        dest="ctrl_args",
        default=ArgList(),
        action=SimulationArgAction,
        prefix="--ctrl.",
        help="Specify arguments for the control script. Use --ctrl. prefix to specify arguments to pass to the control executable."
        "`--ctrl.foo bar` --> `--foo bar`. If single '-' is needed use `--ctrl.-v`. Multiple values "
        "are supported `-opp.bar abc efg 123` will be `--bar abc efg 123`. For possible arguments see help of "
        "executable. Defaults: ",
    )
    parser.add_argument(
        "-wc",
        "--with-control",
        dest="control",
        default="control.py",
        required=True,
        help="Choose file that contains control strategy. (Default: 'control.py')",
    )
    parser.add_argument(
        "--control-use-local",
        dest="ctl_local",
        action="store_true",
        default=False,
        required=False,
        help="If true container uses currently checkout code instead of installed coded during container creation.",
    )


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
        default="latest",
        required=False,
        help="Choose Vadere container. (Default: latest)",
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


def add_omnet_arguments(parser: argparse.ArgumentParser, args: List[str]):
    parser.add_argument(
        "--opp-exec",
        dest="opp_exec",
        default="",
        help="Specify OMNeT++ executable Default($CROWNET_HOME/crownet/src/run_crownet). "
        "Use --opp. prefix to specify arguments to pass to the "
        "given executable.",
    )
    parser.add_argument(
        "--opp.xxx",
        *filter_options(args, "--opp."),
        dest="opp_args",
        default=ArgList.from_list(
            [["-f", "omnetpp.ini"], ["-u", "Cmdenv"], ["-c", "final"]]
        ),
        action=SimulationArgAction,
        prefix="--opp.",
        help="Specify OMNeT++ executable. Use --opp. prefix to specify arguments to pass to the given executable."
        "`--opp.foo bar` --> `--foo bar`. If single '-' is needed use `--opp.-v`. Multiple values "
        "are supported `-opp.bar abc efg 123` will be `--bar abc efg 123`. For possible arguments see help of "
        "executable. Defaults: ",
    )
    parser.add_argument(
        "--omnet-tag",
        dest="omnet_tag",
        default="latest",
        required=False,
        help="Choose Omnet container. (Default: latest)",
    )


def add_sumo_arguments(parser: argparse.ArgumentParser, args: List[str]):
    parser.add_argument(
        "--sumo.xxx",
        *filter_options(args, "--sumo."),
        dest="sumo_args",
        default=ArgList.from_list(
            [
                ["--port", "9999"],
                ["--bind", "0.0.0.0"],
            ]
        ),
        action=SimulationArgAction,
        prefix="--sumo.",
        help="Sumo Arguments",
    )
    parser.add_argument(
        "--create-sumo-container",
        dest="create_sumo_container",
        action="store_true",
        default=False,
        required=False,
        help="If set a sumo container with name sumo_<run-name> is created matching to opp_<run-name> container.",
    )
    parser.add_argument(
        "--sumo-tag",
        dest="sumo_tag",
        default="latest",
        required=False,
        help="Choose Sumo container. (Default: latest)",
    )


def parse_args_as_dict(runner: Any, args=None):
    _args: List[str] = sys.argv[1:] if args is None else args

    # parse arguments
    main: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="BaseRunner", description="Any description"
    )
    parent: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    # arguments used by all sub-commands
    add_base_arguments(parser=parent)

    # subparsers
    sub = main.add_subparsers(title="Available Commands", dest="subparser_name")

    # vadere
    vadere_parser: argparse.ArgumentParser = sub.add_parser(
        "vadere", help="vadere subparser", parents=[parent]
    )
    add_vadere_arguments(parser=vadere_parser)
    vadere_parser.set_defaults(main_func=runner.run_vadere)
    vadere_parser.set_defaults(result_dir_callback=result_dir_vadere_only)

    # vadere control
    vadere_control_parser: argparse.ArgumentParser = sub.add_parser(
        "vadere-control", help="vadere control subparser", parents=[parent]
    )
    add_vadere_arguments(parser=vadere_control_parser)
    add_control_arguments(parser=vadere_control_parser, args=_args)
    vadere_control_parser.set_defaults(main_func=runner.run_simulation_vadere_ctl)
    vadere_control_parser.set_defaults(result_dir_callback=result_dir_vadere_only)

    # vadere omnet
    vadere_opp_parser: argparse.ArgumentParser = sub.add_parser(
        "vadere-opp", help="vadere omnet subparser", parents=[parent]
    )
    add_vadere_arguments(parser=vadere_opp_parser)
    add_omnet_arguments(parser=vadere_opp_parser, args=_args)
    vadere_opp_parser.set_defaults(main_func=runner.run_simulation_omnet_vadere)
    vadere_opp_parser.set_defaults(result_dir_callback=result_dir_with_opp)

    # vadere omnet control
    vadere_opp_control_parser: argparse.ArgumentParser = sub.add_parser(
        "vadere-opp-control", help="vadere omnet control subparser", parents=[parent]
    )
    add_omnet_arguments(parser=vadere_opp_control_parser, args=_args)
    add_vadere_arguments(parser=vadere_opp_control_parser)
    add_control_arguments(parser=vadere_opp_control_parser, args=_args)
    vadere_opp_control_parser.set_defaults(
        main_func=runner.run_simulation_vadere_omnet_ctl
    )
    vadere_opp_control_parser.set_defaults(result_dir_callback=result_dir_with_opp)

    # sumo
    sumo_parser: argparse.ArgumentParser = sub.add_parser(
        "sumo", help="sumo subparser", parents=[parent]
    )
    add_sumo_arguments(parser=sumo_parser, args=_args)
    add_omnet_arguments(sumo_parser, args=_args)
    sumo_parser.set_defaults(main_func=runner.run_simulation_omnet_sumo)
    sumo_parser.set_defaults(result_dir_callback=result_dir_with_opp)

    parsed_args = main.parse_args(_args)
    ns = vars(parsed_args)

    level_idx = ns["verbose"]
    set_level(levels[level_idx])
    set_format("%(asctime)s:%(module)s:%(levelname)s> %(message)s")

    return ns


def result_dir_with_opp(ns, working_dir) -> str:
    """
    set result dir based on OMNeT++
    """
    config = ns["opp_args"].get_value("-c")
    if os.path.abspath(ns["result_dir"]):
        return os.path.join(
            ns["result_dir"],
            f"{config}_{ns['experiment_label']}",
        )
    else:
        return os.path.join(
            working_dir,
            ns["result_dir"],
            f"{config}_{ns['experiment_label']}",
        )


def result_dir_vadere_only(ns, working_dir):
    if os.path.abspath(ns["result_dir"]):
        return ns["result_dir"]
    else:
        return os.path.join(working_dir, ns["result_dir"])


class process_as:
    """
    add priority and type attributes to functions to set dynamic execution order.
    """

    def __init__(self, data):
        self.prio = data["prio"]
        self.type = data["type"]

    def __call__(self, fn):
        fn.__setattr__("prio", self.prio)
        fn.__setattr__("type", self.type)
        return fn


class BaseRunner:
    def __init__(self, working_dir, args=None):
        self.ns = parse_args_as_dict(self, args)
        self.docker_client = DockerClient.get()  # increased timeout
        self.working_dir = working_dir
        self.vadere_runner = None
        self.opp_runner = None
        self.control_runner = None
        self.sumo_runner = None

        # prepare post and pre map
        self.f_map: dict = {}
        for key in [i for i in dir(self) if not i.startswith("__")]:
            __o = self.__getattribute__(key)
            if callable(__o):
                try:
                    __prio = __o.__getattribute__("prio")
                    __type = __o.__getattribute__("type")
                    __type_list = self.f_map.get(__type, [])
                    __type_list.append((__prio, __o))
                    self.f_map[__type] = __type_list
                except AttributeError:
                    continue

    def result_base_dir(self):
        """
        get correct result dir independently of execution setup (opp-vadere, opp-vadere-control, vadere-control, vadere).
        """
        return self.ns["result_dir_callback"](self.ns, self.working_dir)

    def run(self):
        logger.info("execute pre hooks")
        self.pre()
        logger.info("execute simulation")
        ret = self.dispatch_run()
        if ret != 0:
            raise RuntimeError("Error in Simulation")
        logger.info("execute post hooks")
        self.post()
        logger.info("done")

    def sort_processing(self, ptype, method_list):

        map = self.f_map.get(ptype, [])
        method_list = [
            os.path.splitext(qoi)[0].replace("-", "_").lower() for qoi in method_list[0]
        ]
        filtered_map = [
            [prio, _f] for prio, _f in map if _f.__name__.lower() in method_list
        ]
        filtered_map.sort(key=lambda x: x[0], reverse=True)
        return filtered_map

    def post(self):
        method_list = self.ns["qoi"]
        if method_list:
            _post_map = self.sort_processing("post", method_list)
            for prio, _f in _post_map:
                print(f"post: '{_f.__name__}' as post function with prio: {prio} ...")
                _f()

    def pre(self):
        method_list = self.ns["pre"]
        if method_list:
            _pre_map = self.sort_processing("pre", method_list)
            for prio, _f in _pre_map:
                print(f"pre: '{_f.__name__}' as post function with prio: {prio} ...")
                _f()

    def build_opp_runner(self):
        run_name = self.ns["run_name"]
        # set default executable based on $CRWNET_HOME variable
        opp_exec = (
            self.ns["opp_exec"]
            if bool(self.ns["opp_exec"])
            else CrowNetConfig.join_home(f"crownet/src/run_crownet")
        )
        self.opp_runner = OppRunner(
            docker_client=self.docker_client,
            name=f"omnetpp_{run_name}",
            tag=self.ns["omnet_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=False,  # do not detach --> wait on opp container
            journal_tag=f"omnetpp_{run_name}",
            run_cmd=opp_exec,
        )
        self.opp_runner.apply_reuse_policy()
        self.opp_runner.set_working_dir(self.working_dir)
        if self.ns["write_container_log"]:
            self.opp_runner.set_log_callback(
                ContainerLogWriter(f"{self.result_base_dir()}/container_opp.out")
            )

    def build_sumo_runner(self):
        run_name = self.ns["run_name"]
        self.sumo_runner = SumoRunner(
            docker_client=self.docker_client,
            name=f"sumo_{run_name}",
            tag=self.ns["sumo_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=True,  # we do not wait for this container (see OMNeT container)
            journal_tag=f"sumo_{run_name}",
        )
        self.sumo_runner.apply_reuse_policy()
        self.sumo_runner.set_working_dir(self.working_dir)
        if self.ns["write_container_log"]:
            self.sumo_runner.set_log_callback(
                ContainerLogWriter(f"{self.result_base_dir()}/container_sumo.out")
            )

    def build_and_start_control_runner(self):
        self.build_control_runner(detach=True)
        self.exec_control_runner(mode="server")

    def build_and_start_vadere_runner(self, port=None, output_dir=None):

        if port is None:
            port = self.ns["v_traci_port"]

        run_name = self.ns["run_name"]
        self.vadere_runner = VadereRunner(
            docker_client=self.docker_client,
            name=f"vadere_{run_name}",
            tag=self.ns["vadere_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=True,  # detach at first and wait vadere container after opp container is done
            journal_tag=f"vadere_{run_name}",
        )
        self.vadere_runner.apply_reuse_policy()
        self.vadere_runner.set_working_dir(self.working_dir)
        if self.ns["write_container_log"]:
            self.vadere_runner.set_log_callback(
                ContainerLogWriter(f"{self.result_base_dir()}/container_vadere.out")
            )

        logfile = os.devnull
        if self.ns["v_logfile"] != "":
            logfile = self.ns["v_logfile"]

        # start vadere container detached in the background. Will be stoped in the finally block
        self.vadere_runner.exec_single_server(
            traci_port=port,
            loglevel=self.ns["v_loglevel"],
            logfile=logfile,
            output_dir=output_dir,
        )

    def build_control_runner(self, detach=False):

        run_name = self.ns["run_name"]
        self.control_runner = ControlRunner(
            docker_client=self.docker_client,
            name=f"control_{run_name}",
            tag=self.ns["control_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=detach,  # do not detach --> wait on control container
            journal_tag=f"control_{run_name}",
        )
        self.control_runner.apply_reuse_policy()
        if self.ns["write_container_log"]:
            self.control_runner.set_log_callback(
                ContainerLogWriter(f"{self.result_base_dir()}/container_control.out")
            )

    def exec_control_runner(self, mode):

        if mode == "client":

            host_name = f"vadere_{self.ns['run_name']}"
            experiment_label = (
                f"{ControlRunner.OUTPUT_DEFAULT}_{self.ns['experiment_label']}"
            )

            _wait_for_vadere = True
            while _wait_for_vadere:
                for container in docker.from_env().containers.list():
                    if container.name == host_name:
                        time.sleep(1)
                        _wait_for_vadere = False

            return self.control_runner.start_controller(
                control_file=self.ns["control"],
                host_name=host_name,
                connection_mode="client",
                traci_port=9999,
                use_local=self.ns["ctl_local"],
                scenario=self.ns["scenario_file"],
                ctrl_args=self.ns["ctrl_args"],
                result_dir=self.ns["result_dir"],
                experiment_label=experiment_label,
            )
        else:

            client_name = f"control_{self.ns['run_name']}"

            return self.control_runner.start_controller(
                control_file=self.ns["control"],
                host_name=client_name,
                connection_mode="server",
                traci_port=9997,
                use_local=self.ns["ctl_local"],
                ctrl_args=self.ns["ctrl_args"],
            )

    def build_and_start_vadere_only(self):
        run_name = self.ns["run_name"]
        self.vadere_runner = VadereRunner(
            docker_client=self.docker_client,
            name=f"vadere_{run_name}",
            tag=self.ns["vadere_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=False,  # detach at first and wait vadere container after opp container is done
            journal_tag=f"vadere_{run_name}",
        )
        self.vadere_runner.apply_reuse_policy()
        self.vadere_runner.set_working_dir(self.working_dir)
        if self.ns["write_container_log"]:
            self.vadere_runner.set_log_callback(
                ContainerLogWriter(f"{self.result_base_dir()}/container_vadere.out")
            )

        os.makedirs(self.result_base_dir(), exist_ok=True)

        # start vadere container detached in the background. Will be stoped in the finally block
        self.vadere_runner.exec_vadere_only(
            scenario_file=self.ns["scenario_file"], output_path=self.result_base_dir()
        )

    @staticmethod
    def wait_for_file(filepath, timeout_sec=120):
        sec = 0
        while not os.path.exists(filepath):
            time.sleep(1)
            sec += 1
            if sec >= timeout_sec:
                raise TimeoutError(f"Timeout reached while waiting for {filepath}")
        return filepath

    def dispatch_run(self):
        main_func = self.ns["main_func"]
        ret = main_func()
        return ret

    def run_vadere(self):

        ret = 255
        logger.info("Run vadere in container")

        try:
            self.build_and_start_vadere_only()
            ret = 0  # all good if we reached this.

        except RuntimeError as cErr:
            logger.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logger.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise

        finally:
            # always stop container and delete if no error occurred
            logger.debug(f"cleanup with ret={ret}")

            # TODO: does not work

            # if self.vadere_runner is not None:
            #     self.vadere_runner.container_cleanup(has_error_state=err_state)
            # self.opp_runner.container_cleanup(has_error_state=err_state)

        return ret

    def run_simulation_omnet_sumo(self):
        ret = 255  # fail
        self.build_opp_runner()

        try:
            sumo_args = self.ns["sumo_args"]
            if self.ns["create_sumo_container"]:
                self.build_sumo_runner()
                self.sumo_runner.single_launcher(
                    traci_port=sumo_args.get_value("--port"),
                    bind=sumo_args.get_value("--bind"),
                )

            if self.ns["override-host-config"]:
                self.ns["opp_args"].add_override(
                    f"--sumo-host={self.sumo_runner.name}:{sumo_args.get_value('--port')}"
                )

            # start OMNeT++ container and attach to it
            logger.info(f"start simulation {self.ns['run_name']} ...")
            opp_ret = self.opp_runner.exec_opp_run(
                arg_list=self.ns["opp_args"],
                result_dir=self.ns["result_dir"],
                experiment_label=self.ns["experiment_label"],
                run_args_override={},
            )
            ret = opp_ret["StatusCode"]
            if ret != 0:
                raise RuntimeError(f"OMNeT++ container exited with StatusCode '{ret}'")

            if self.sumo_runner is not None:
                try:
                    self.sumo_runner.container.wait(timeout=600)
                except ReadTimeout:
                    logger.error(
                        f"Timeout (60s) reached while waiting for sumo container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logger.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logger.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logger.debug(f"cleanup with ret={ret}")
            if self.sumo_runner is not None:
                self.sumo_runner.container_cleanup(has_error_state=err_state)
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret

    def run_simulation_omnet_vadere(self):
        ret = 255
        self.build_opp_runner()

        try:
            if self.ns["create_vadere_container"]:
                self.build_and_start_vadere_runner()
            if self.ns["override-host-config"]:
                self.ns["opp_args"].add(f"--vadere-host={self.vadere_runner.name}")
            # start OMNeT++ container and attach to it.
            logger.info(f"start simulation {self.ns['run_name']} ...")
            opp_ret = self.opp_runner.exec_opp_run(
                arg_list=self.ns["opp_args"],
                result_dir=self.ns["result_dir"],
                experiment_label=self.ns["experiment_label"],
                run_args_override={},
            )

            ret = opp_ret["StatusCode"]
            if ret != 0:
                raise RuntimeError(f"OMNeT++ container exited with StatusCode '{ret}'")

            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])
                except ReadTimeout:
                    logger.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logger.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logger.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logger.debug(f"cleanup with ret={ret}")
            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret

    def run_simulation_vadere_omnet_ctl(self):

        logger.info(
            "Control vadere with omnetpp. Client 1: omnet, server 1: vadere, port: 9998, Client 2: omnet, server 2: controller, port: 9997"
        )

        ret = 255
        self.build_opp_runner()

        try:
            if self.ns["create_vadere_container"]:
                self.build_and_start_vadere_runner()

            self.build_and_start_control_runner()
            if self.ns["override-host-config"]:
                self.ns["opp_args"].add(f"--vadere-host={self.vadere_runner.name}")
                self.ns["opp_args"].add(f"--flow-host={self.control_runner.name}")

            # start OMNeT++ container and attach to it.
            logger.info(f"start simulation {self.ns['run_name']} ...")
            opp_ret = self.opp_runner.exec_opp_run(
                arg_list=self.ns["opp_args"],
                result_dir=self.ns["result_dir"],
                experiment_label=self.ns["experiment_label"],
                run_args_override={},
            )
            ret = opp_ret["StatusCode"]
            if ret != 0:
                raise RuntimeError(f"OMNeT++ container exited with StatusCode '{ret}'")

            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])
                except ReadTimeout:
                    logger.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255
            if self.control_runner is not None:
                try:
                    self.control_runner.container.wait(
                        timeout=self.ns["v_wait_timeout"]
                    )
                except ReadTimeout:
                    logger.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for controler container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logger.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logger.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logger.debug(f"cleanup with ret={ret}")
            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)
            if self.control_runner is not None:
                self.control_runner.container_cleanup(has_error_state=err_state)
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret

    def run_simulation_vadere_ctl(self):

        ret = 255
        logger.info(
            "Control vadere without omnetpp. Client: controller, server: vadere, port: 9999"
        )

        self.build_control_runner()

        try:
            if self.ns["create_vadere_container"]:
                self.build_and_start_vadere_runner(port=9999)
                logger.info(f"start simulation {self.ns['run_name']} ...")

            ctl_ret = self.exec_control_runner(mode="client")
            ret = ctl_ret["StatusCode"]
            if ret != 0:
                raise RuntimeError(f"Control container exited with StatusCode '{ret}'")

            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])
                except ReadTimeout:
                    logger.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255
        except RuntimeError as cErr:
            logger.error(cErr)
            ret = 255
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logger.debug(f"cleanup with ret={ret}")
            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)
            if self.control_runner is not None:
                self.control_runner.container_cleanup(has_error_state=err_state)
        return ret


if __name__ == "__main__":
    b = BaseRunner(".")
    print("hi")
