import argparse
import logging
import os
import signal
import time
from datetime import datetime

import docker
from requests.exceptions import ReadTimeout

from roveranalyzer.dockerrunner.dockerrunner import DockerCleanup, DockerReuse
from roveranalyzer.simulators.controller.controllerrunner import ControlRunner
from roveranalyzer.simulators.opp.runner import OppRunner
from roveranalyzer.simulators.vadere.runner import VadereRunner

if len(logging.root.handlers) == 0:
    # set logger for dev (will be overwritten if needed)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(module)s:%(levelname)s> %(message)s",
    )

# todo: split in default and simulator specific arguments
def parse_args_as_dict(args=None):

    # parse arguments
    parser = argparse.ArgumentParser()
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
        "-i",
        "--ini-file",
        dest="opp_ini",
        default="omnetpp.ini",
        required=False,
        help="Ini-file for simulation. Default: omnetpp.ini",
    )

    parser.add_argument(
        "-sf",
        "--scenario-file",
        dest="scenario_file",
        default="",
        required=False,
        help="Scenario-file *.scenario for Vadere simulation.",
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        default="final",
        required=False,
        help="Config to simulation. Default: final",
    )
    parser.add_argument(
        "--resultdir",
        dest="result_dir",
        default="results",
        required=False,
        help="Result directory. Default: results",
    )
    parser.add_argument(
        "--experiment-label",
        dest="experiment_label",
        default="out",
        required=False,
        help="experiment-label used in the result path. Default: out",
    )
    parser.add_argument(
        "--use-timestep-label",
        dest="use_timestep_label",
        default=False,
        required=False,
        action="store_true",
        help="Use current timestamp (sanitized ISO-Format). If this is given '--experiment-label' will be ignored. "
        "Default: False",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        nargs="?",
        default="rover_run",
        help="Set name of current run. This will be CONTAINER_TAG for journald. Default: rover_run",
    )

    parser.add_argument(
        "--delete-existing-containers",
        dest="delete_existing_containers",
        action="store_true",
        default=False,
        required=False,
        help="Delete existing (stopped) containers with the same name.",
    )

    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        required=False,
        action="store_true",
        help="Use opp_run_debug Default: False",
    )

    # ToDo: not possible any more.
    parser.add_argument(
        "--run-all",
        dest="run_all",
        default=False,
        required=False,
        action="store_true",
        help="Use OMNeT++ internal parameter variation. Not compatible with --debug. Default: False",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        dest="jobs",
        default=-1,
        required=False,
        help="In conjunction with --run-all. Set number of parallel executions. Default: Number of Cores.",
    )
    parser.add_argument(
        "--log-journald",
        dest="log_journald",
        action="store_true",
        default=True,
        required=False,
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
        "--create-log-file",
        dest="create_log_file",
        action="store_true",
        default=False,
        required=False,
        help="Redirect log messages to Logfile at script location.",
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
        "--verbose",
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Set verbosity of command. From warnings and errors only (-v) to debug output (-vvv)",
    )

    parser.add_argument(
        "--silent",
        "-s",
        dest="silent",
        action="store_true",
        default=False,
        required=False,
        help="No output is generated. Only fatal errors leading to non zero exit codes.",
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
        "--omnet-tag",
        dest="omnet_tag",
        default="latest",
        required=False,
        help="Choose Omnet container. (Default: latest)",
    )

    parser.add_argument(
        "--control-tag",
        dest="control_tag",
        default="latest",
        required=False,
        help="Choose Control container. (Default: latest)",
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

    parser.add_argument(
        "-wc",
        "--with-control",
        dest="control",
        default=None,
        required=False,
        help="Choose file that contains control strategy. (Default: '')",
    )

    parser.add_argument(
        "--control-vadere-only",
        dest="control_vadere_only",
        action="store_true",
        default=False,
        required=False,
        help="If set the control action is applied without omnetpp. Direct information dissemination without delay.",
    )

    parser.add_argument(
        "--vadere-only",
        dest="vadere_only",
        action="store_true",
        default=False,
        required=False,
        help="If set run Vadere in container without omnetpp or control.",
    )

    if args is None:
        ns = vars(parser.parse_args())
    else:
        ns = vars(parser.parse_args(args))

    if ns["use_timestep_label"]:
        ns["experiment_label"] = (
            datetime.now().isoformat().replace("-", "").replace(":", "")
        )

    # remove existing handlers and overwrite with user settings
    for h in logging.root.handlers:
        logging.root.removeHandler(h)

    levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]
    if ns["silent"]:
        level_idx = 0
    else:
        level_idx = ns["verbose"]
    if ns["create_log_file"]:
        logging.basicConfig(
            level=levels[level_idx],
            format="%(asctime)s:%(module)s:%(levelname)s> %(message)s",
            filename=f"{os.getcwd()}/runner.log",
        )
    else:
        logging.basicConfig(
            level=levels[level_idx],
            format="%(asctime)s:%(module)s:%(levelname)s> %(message)s",
        )
    return ns


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

        self.ns = parse_args_as_dict(args)
        self.docker_client = docker.from_env()
        self.working_dir = working_dir
        self.vadere_runner = None
        self.opp_runner = None
        self.control_runner = None

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

    def run(self):
        logging.debug("execute pre hooks")
        self.pre()
        logging.debug("execute simulation")
        ret = self.run_simulation()
        if ret != 0:
            raise RuntimeError("Error in Simulation")
        logging.debug("execute post hooks")
        self.post()

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

    def build_opp_runer(self):
        run_name = self.ns["run_name"]
        self.opp_runner = OppRunner(
            docker_client=self.docker_client,
            name=f"omnetpp_{run_name}",
            tag=self.ns["omnet_tag"],
            cleanup_policy=self.ns["cleanup_policy"],
            reuse_policy=self.ns["reuse_policy"],
            detach=False,  # do not detach --> wait on opp container
            journal_tag=f"omnetpp_{run_name}",
            debug=self.ns["debug"],
        )
        self.opp_runner.apply_reuse_policy()
        self.opp_runner.set_working_dir(self.working_dir)

    def build_and_start_control_runner(self, port=9997):
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

    def exec_control_runner(self, mode):

        if mode == "client":

            host_name = f"vadere_{self.ns['run_name']}"

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
            )
        else:

            client_name = f"control_{self.ns['run_name']}"

            return self.control_runner.start_controller(
                control_file=self.ns["control"],
                host_name=client_name,
                connection_mode="server",
                traci_port=9997,
            )

    def is_controlled(self):
        if self.ns["control"] is None:
            return False
        else:
            return True

    def is_control_vadere_directly(self):

        if self.is_controlled():
            return self.ns["control_vadere_only"]
        else:
            raise ValueError("Control file not set.")

    def run_simulation(self):

        if self.is_controlled():
            return self.run_simulation_controlled()
        else:
            return self.run_simulation_uncontrolled()

    def run_simulation_controlled(self):

        if self.is_control_vadere_directly():
            ret = self.run_simulation_vadere_ctl_only()
        else:
            ret = self.run_simulation_vadere_omnet_ctl()

        return ret

    def run_simulation_vadere_ctl_only(self):

        ret = 255
        logging.info(
            "Control vadere without omnetpp. Client: controller, server: vadere, port: 9999"
        )

        output_dir = os.path.join(
            os.getcwd(),
            f"results/vadere_controlled_{self.ns['experiment_label']}/vadere.d",
        )
        os.makedirs(output_dir, exist_ok=True)

        self.build_control_runner()

        try:
            if self.ns["create_vadere_container"]:

                self.build_and_start_vadere_runner(port=9999, output_dir=output_dir)
                logging.info(f"start simulation {self.ns['run_name']} ...")

            ret_control, control_container = self.exec_control_runner(mode="client")

            ret = 0  # all good if we reached this.
            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])

                except ReadTimeout:
                    logging.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logging.error(cErr)
            ret = 255

        except KeyboardInterrupt as K:
            logging.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise

        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logging.debug(f"cleanup with ret={ret}")

            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)

            self.control_runner.container_cleanup(has_error_state=err_state)

        return ret

    def run_simulation_uncontrolled(self):

        if self.ns["vadere_only"]:
            return self.run_vadere()
        else:
            return self.run_simulation_uncontrolled_crownet()

    def build_and_start_vadere_only(self, port=None):

        if port is None:
            port = self.ns["v_traci_port"]

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

        logfile = os.devnull
        if self.ns["v_logfile"] != "":
            logfile = self.ns["v_logfile"]

        result_path = os.path.join(
            os.getcwd(), f"results/vadere_only_{self.ns['experiment_label']}/vadere.d"
        )

        os.makedirs(result_path, exist_ok=True)

        # start vadere container detached in the background. Will be stoped in the finally block
        self.vadere_runner.exec_vadere_only(
            scenario_file=self.ns["scenario_file"], output_path=result_path
        )

    def run_vadere(self):

        ret = 255
        logging.info("Run vadere in container")

        try:
            self.build_and_start_vadere_only()
            ret = 0  # all good if we reached this.

        except RuntimeError as cErr:
            logging.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logging.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise

        finally:
            # always stop container and delete if no error occurred
            err_state = ret
            logging.debug(f"cleanup with ret={ret}")

            # TODO: does not work

            # if self.vadere_runner is not None:
            #     self.vadere_runner.container_cleanup(has_error_state=err_state)
            # self.opp_runner.container_cleanup(has_error_state=err_state)

        return ret

    def run_simulation_uncontrolled_crownet(self):

        ret = 255
        self.build_opp_runer()

        try:

            if self.ns["create_vadere_container"]:
                self.build_and_start_vadere_runner()

            # start OMNeT++ container and attach to it.
            logging.info(f"start simulation {self.ns['run_name']} ...")
            ret_opp, opp_container = self.opp_runner.exec_opp_run(
                **self.ns, run_args_override={}
            )

            ret = 0  # all good if we reached this.
            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])
                except ReadTimeout:
                    logging.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logging.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logging.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logging.debug(f"cleanup with ret={ret}")
            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret

    def result_base_dir(self):
        """
        returns base path for output. Structure is based on OMNeT++ default.
        ${resultdir}/${configname}_${experiment}/.....
        """
        return os.path.join(
            self.working_dir,
            self.ns["result_dir"],
            f"{self.ns['config']}_{self.ns['experiment_label']}",
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

    def run_simulation_vadere_omnet_ctl(self):

        logging.info(
            "Control vadere with omnetpp. Client 1: omnet, server 1: vadere, port: 9998, Client 2: omnet, server 2: controller, port: 9997"
        )

        ret = 255
        self.build_opp_runer()

        try:
            if self.ns["create_vadere_container"]:
                self.build_and_start_vadere_runner()

            self.build_and_start_control_runner()

            # start OMNeT++ container and attach to it.
            logging.info(f"start simulation {self.ns['run_name']} ...")
            ret_opp, opp_container = self.opp_runner.exec_opp_run(
                **self.ns, run_args_override={}
            )

            ret = 0  # all good if we reached this.
            if self.vadere_runner is not None:
                try:
                    self.vadere_runner.container.wait(timeout=self.ns["v_wait_timeout"])
                except ReadTimeout:
                    logging.error(
                        f"Timeout ({self.ns['v_wait_timeout']}) reached while waiting for vadere container to finished"
                    )
                    ret = 255

        except RuntimeError as cErr:
            logging.error(cErr)
            ret = 255
        except KeyboardInterrupt as K:
            logging.info("KeyboardInterrupt detected. Shutdown. ")
            ret = 128 + signal.SIGINT
            raise
        finally:
            # always stop container and delete if no error occurred
            err_state = ret != 0
            logging.debug(f"cleanup with ret={ret}")
            if self.vadere_runner is not None:
                self.vadere_runner.container_cleanup(has_error_state=err_state)
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret


if __name__ == "__main__":

    b = BaseRunner(".")
    print("hi")
