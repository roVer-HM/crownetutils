import os
import subprocess
import sys
from pathlib import Path

import docker
from docker.errors import ContainerError
from docker.types import LogConfig
from oppanalyzer.configuration import RoverConfig


class DockerRunner:

    NET = "rovernet"
    _default_enviroment = {}

    def __init__(self, image, tag="latest", docker_client=None):
        if docker_client is None:
            self.client: docker.DockerClient = docker.from_env()
        else:
            self.client: docker.DockerClient = docker_client
        self.image = f"{image}:{tag}"
        self.user_home = str(Path.home())
        self.user_id = os.getuid()
        self.working_dir = str(Path.cwd())
        self.name = ""
        self.hostname = ""

        # last call in init.
        self._apply_default_volumes()
        self._apply_default_environment()
        self._apply_default_run_args()

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

    def set_runargs(self, key, value):
        self.run_args[key] = value

    def set_working_dir(self, wdir):
        self.run_args["working_dir"] = wdir

    def _apply_default_environment(self):
        self.environment: dict = {}
        if "DISPLAY" in os.environ:
            self.environment["DISPLAY"] = os.environ.get("DISPLAY")

    def check_create_network(self):
        existing_networks = [n.name for n in self.client.networks.list()]
        if self.NET not in existing_networks:
            self.client.networks.create(self.NET)
        return self.NET

    def _apply_default_run_args(self):
        self.run_args: dict = {
            "cap_add": ["SYS_PTRACE"],
            "user": self.user_id,
            "network": self.check_create_network(),
            "environment": self.environment,
            "volumes": self.volumes,
            "working_dir": self.working_dir,
            "detach": True,
        }

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

    def run(self, cmd="/init.sh", **run_args):
        self.build_run_args(**run_args)
        print("run container")
        command = self.wrap_command(cmd)
        return self.client.containers.run(
            image=self.image, command=command, **self.run_args
        )


class OppRunner(DockerRunner):
    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/rover-main/omnetpp",
        tag="latest",
        docker_client=None,
    ):
        super().__init__(image, tag, docker_client)

    def _apply_default_environment(self):
        super()._apply_default_environment()
        nedpath = (
            subprocess.check_output(f"{os.environ['ROVER_MAIN']}/scripts/nedpath")
            .decode("utf-8")
            .strip()
        )
        self.environment["NEDPATH"] = nedpath

    @staticmethod
    def __build_base_cmd(base_cmd):
        cmd = [base_cmd]
        cmd.extend(["-u", "Cmdenv"])
        cmd.extend(["-l", RoverConfig.join_rover_main("inet4/src/INET")])
        cmd.extend(["-l", RoverConfig.join_rover_main("rover/src/ROVER")])
        cmd.extend(["-l", RoverConfig.join_rover_main("simulte/src/lte")])
        cmd.extend(["-l", RoverConfig.join_rover_main("veins/src/veins")])
        cmd.extend(
            [
                "-l",
                RoverConfig.join_rover_main(
                    "veins/subprojects/veins_inet/src/veins_inet"
                ),
            ]
        )
        return cmd

    def opp_query_details(
        self,
        opp_ini="omnetpp.ini",
        config="final",
        result_dir="results",
        experiment_label="out",
        run_args_override=None,
        **kwargs,
    ):
        cmd = self.__build_base_cmd("opp_run")
        cmd.extend(["-c", config])
        if experiment_label is not None:
            cmd.extend([f"--experiment-label={experiment_label}"])
        cmd.extend([f"--result-dir={result_dir}"])
        cmd.extend(["-q", "rundetails"])
        cmd.append(opp_ini)

        return self.run(cmd, **run_args_override)

    def opp_run(
        self,
        opp_ini="omnetpp.ini",
        config="final",
        result_dir="results",
        experiment_label="out",
        debug=False,
        run_all=True,
        jobs=-1,
        run_args_override=None,
        **kwargs,
    ):
        if run_all and debug:
            raise ValueError("run_all and debug not supported")
        cmd = "opp_run"
        if debug:
            cmd = f"{cmd}_dbg"
        cmd = self.__build_base_cmd(cmd)
        cmd.extend(["-c", config])
        if experiment_label is not None:
            cmd.extend([f"--experiment-label={experiment_label}"])
        cmd.extend([f"--result-dir={result_dir}"])
        cmd.append(opp_ini)

        if run_all:
            cmd.insert(0, "opp_runall")
            if jobs > 0:
                cmd.insert(1, "-j")
                cmd.insert(2, str(jobs))

        return self.run(cmd, **run_args_override)


class VadereRunner(DockerRunner):
    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/rover-main/vadere",
        tag="latest",
        docker_client=None,
    ):
        super().__init__(image, tag, docker_client)
        self.name = "vadere"

    def _apply_default_volumes(self):
        super()._apply_default_volumes()
        self.volumes.update({})

    def _apply_default_environment(self):
        super()._apply_default_environment()
        self.environment.update(
            {
                "TRACI_PORT": 9998,
                "TRACI_GUI": "false",
                # "TRACI_DEBUG": "false",
                # "VADERE_LOG_LEVEL": "INFO",
                # "VADERE_LOG": self.working_dir,
            }
        )

    def _apply_default_run_args(self):
        super()._apply_default_run_args()
        self.run_args["remove"] = True


if __name__ == "__main__":
    client = docker.from_env()
    os.chdir("/home/sts/repos/rover-main/rover/simulations/mucFreiNetdLTE2dMulticast")
    opp = OppRunner(docker_client=client)
    log = LogConfig(type=LogConfig.types.JOURNALD, config={"tag": "opp_01"})
    opp.set_log_driver(log)
    try:
        r = opp.opp_run(
            opp_ini="omnetpp.ini",
            result_dir="result2",
            config="vadere01",
            debug=True,
            run_all=False,
            run_args_override={"detach": False},
        )
        print(r.decode("utf-8"))
    except ContainerError as cErr:
        print(
            f"Error in container '{cErr.container.name}' exit_status: {cErr.exit_status}",
            file=sys.stderr,
        )
        print(f"\tImage: {cErr.image}", file=sys.stderr)
        print(f"\tCommand: {cErr.command}", file=sys.stderr)
        print(f"\tstderr:", file=sys.stderr)
        err_str = cErr.stderr.decode("utf-8").strip().split("\n")
        for line in err_str:
            print(f"\t{line}", )
