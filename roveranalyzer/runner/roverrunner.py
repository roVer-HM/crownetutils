import argparse
import logging
import os
import sys
import time
from datetime import datetime

import docker
from requests.exceptions import ReadTimeout

from roveranalyzer.runner.dockerrunner import OppRunner, VadereRunner

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s:%(filename)s:%(levelname)s> %(message)s"
)


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
        # TODO rename
        "--use-timestep-label",
        dest="use_timestep_label",
        default=False,
        required=False,
        action="store_true",
        help="Use current timestamp (sanitized ISI-Format). If this is given '--experiment-label' will be ignored. "
        "Default: False",
    )
    parser.add_argument(
        "--run-name",
        dest="run_name",
        nargs="?",
        default="opp_run",
        help="Set name of current run. This will be CONTAINER_TAG for journald. Default: opp_run",
    )
    parser.add_argument(
        "--debug",
        dest="debug",
        default=False,
        required=False,
        action="store_true",
        help="Use opp_run_debug Default: False",
    )
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
        "--keep_container",
        dest="keep_container",
        action="store_true",
        default=False,
        required=False,
        help="If set the container is not NOT deleted after execution. This simplifies debugging.",
    )

    parser.add_argument(
        "--v.traci-port",
        dest="v_traci_port",
        default="9998",
        required=False,
        help="Set TraCI Port in Vadere container. (Default: 9998)",
    )

    parser.add_argument(
        "--v.loglevel",
        dest="v_loglevel",
        default="INFO",
        required=False,
        help="Set loglevel of TraCI Server [WARN, INFO, DEBUG, TRACE]. (Default: INFO)",
    )

    parser.add_argument(
        "--v.logfile",
        dest="v_logfile",
        default="",
        required=False,
        help="Set log file name. If not set '', log file will not be created. "
        "This setting has no effect on --log-journald. (Default: '') ",
    )

    if args is None:
        ns = vars(parser.parse_args())
    else:
        ns = vars(parser.parse_args(args))

    if ns["use_timestep_label"]:
        ns["experiment_label"] = (
            datetime.now().isoformat().replace("-", "").replace(":", "")
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
        self.docker_client = docker.from_env()
        self.ns = parse_args_as_dict(args)
        self.working_dir = working_dir

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
        self.pre()
        self.run_simulation()
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

    def run_simulation(self):
        run_name = self.ns["run_name"]
        journal_tag = ""
        if self.ns["log_journald"]:
            journal_tag = run_name

        opp_runner = OppRunner(
            docker_client=self.docker_client,
            name=f"omnetpp_{run_name}",
            remove=True,
            detach=False,  # do not detach --> wait on opp container
            journal_tag=f"omnetpp_{journal_tag}",
        )
        opp_runner.delete_if_container_exists()
        opp_runner.set_working_dir(self.working_dir)

        vadere_runner = VadereRunner(
            docker_client=self.docker_client,
            name=f"vadere_{run_name}",
            remove=True,
            detach=True,  # detach at first and wait vadere container after opp container is done
            journal_tag=f"vadere_{journal_tag}",
        )
        vadere_runner.delete_if_container_exists()
        opp_runner.set_working_dir(self.working_dir)

        try:
            logfile = os.devnull
            if self.ns["v_logfile"] != "":
                logfile = self.ns["v_logfile"]

            ret_vadere, vadere_container = vadere_runner.exec_single_server(
                traci_port=self.ns["v_traci_port"],
                loglevel=self.ns["v_loglevel"],
                logfile=logfile,
            )

            # todo: check if vadere_container is running
            # while:....

            # todo: check if namespace cleanup is needed due to v_***
            ret_opp, opp_contaier = opp_runner.exec_opp_run(
                **self.ns, run_args_override={}
            )

            # todo: wait for vadere container to reach exit state
            try:
                vadere_container.wait(timeout=180)
            except ReadTimeout as err:
                logging.error(
                    f"Timeout reached while waiting for vadere container to finsh"
                )
                vadere_container.stop()
                opp_contaier.stop()

        except RuntimeError as cErr:
            logging.error(cErr)
            # self.__print_err(cErr)
            sys.exit(-1)

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

    # def __print_err(self, cErr):
    #     print(
    #         f"Error in container '{cErr.container.name}' exit_status: {cErr.exit_status}",
    #         file=sys.stderr,
    #     )
    #     print(f"\tImage: {cErr.image}", file=sys.stderr)
    #     print(f"\tCommand: {cErr.command}", file=sys.stderr)
    #     print(f"\tstderr:", file=sys.stderr)
    #     err_str = cErr.stderr.decode("utf-8").strip().split("\n")
    #     for line in err_str:
    #         print(f"\t{line}", file=sys.stderr)
    #     if self.ns["log_journald"]:
    #         print(
    #             f'For full container output see: journalctl -b CONTAINER_TAG={self.ns["run_name"]} --all',
    #             file=sys.stderr,
    #         )


if __name__ == "__main__":
    runner = BaseRunner(os.getcwd())
    runner.run()
