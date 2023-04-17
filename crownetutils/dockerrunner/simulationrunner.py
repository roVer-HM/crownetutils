import os
import signal
import time
from typing import Any, List

import docker
from requests.exceptions import ReadTimeout

from crownetutils.dockerrunner.dockerrunner import ContainerLogWriter, DockerClient
from crownetutils.dockerrunner.run_argparser import parse_run_script_arguments
from crownetutils.dockerrunner.simulators.controllerrunner import ControlRunner
from crownetutils.dockerrunner.simulators.omnetrunner import OppRunner
from crownetutils.dockerrunner.simulators.sumorunner import SumoRunner
from crownetutils.dockerrunner.simulators.vadererunner import VadereRunner
from crownetutils.utils.logging import logger
from crownetutils.utils.path import PathHelper


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


class BaseSimulationRunner:
    @classmethod
    def from_config(cls, workding_dir, config_path):
        return cls(workding_dir, args=["config", "-f", config_path])

    def __init__(self, working_dir, args=None):
        self.ns = parse_run_script_arguments(self, args)
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

    def result_dir(self, *paths):
        return os.path.join(self.result_base_dir(), *paths)

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

    @staticmethod
    def _to_int_if_possible(qoi: List[str]) -> List[Any]:
        ret_str = []
        ret_int = []
        for q in qoi:
            try:
                ret_int.append(int(q))
            except Exception:
                ret_str.append(q)
        return ret_str, ret_int

    def sort_processing(self, ptype, method_list):
        map = self.f_map.get(ptype, [])
        method_list = [
            os.path.splitext(qoi)[0].replace("-", "_").lower() for qoi in method_list[0]
        ]
        method_list_str, method_list_int = self._to_int_if_possible(method_list)
        if len(method_list_str) > 0 and method_list_str[0] == "all":
            # if all is used all other mentioned values are seen as 'remove' filter
            _map = [m for m in map if m[1].__name__.lower() not in method_list_str[1:]]
            _map = [m for m in map if m[0] not in method_list_int]
            method_list = [_f.__name__.lower() for _, _f in _map]
        else:
            method_list = []
            _map = []
            if len(method_list_str) > 0:
                _map.extend(
                    [m for m in map if m[1].__name__.lower() in method_list_str]
                )
            if len(method_list_int) > 0:
                _map.extend([m for m in map if m[0] in method_list_int])
            method_list = list(set([_f.__name__.lower() for _, _f in _map]))

        filtered_map = [
            [prio, _f] for prio, _f in map if _f.__name__.lower() in method_list
        ]
        filtered_map.sort(key=lambda x: x[0], reverse=True)
        return filtered_map

    def post(self):
        method_list = self.ns["qoi"]
        err = []
        if method_list:
            _post_map = self.sort_processing("post", method_list)
            for prio, _f in _post_map:
                print(f"post: '{_f.__name__}' as post function with prio: {prio} ...")
                try:
                    _f()
                except Exception as e:
                    _err = f"Error while executing post processing {prio}:{_f.__name__}>> {e}"
                    logger.error(_err)
                    logger.error(e.print_exc())
                    err.append(f"  {_err}")
                    break

            if len(err) > 0:
                err = "\n".join(err)
                raise RuntimeError(f"Error in Postprocessing:\n{err}")

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
            else PathHelper.crownet_home().join("crownet/src/run_crownet")
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

    def run_postprocessing_only(self) -> int:
        # do nothing only run set postprocessing
        return 0

    def run_vadere(self) -> int:
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

    def run_simulation_omnet(self) -> int:
        ret = 255  # fail
        self.build_opp_runner()

        try:
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
            self.opp_runner.container_cleanup(has_error_state=err_state)
        return ret

    def run_simulation_omnet_sumo(self) -> int:
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

    def run_simulation_omnet_vadere(self) -> int:
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

    def run_simulation_vadere_omnet_ctl(self) -> int:
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

    def run_simulation_vadere_ctl(self) -> int:
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
