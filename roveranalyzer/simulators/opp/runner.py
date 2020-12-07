import os
import subprocess

from roveranalyzer.dockerrunner.dockerrunner import DockerRunner
from roveranalyzer.simulators.opp.configuration import RoverConfig


class OppRunner(DockerRunner):
    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/rover-main/omnetpp",
        tag="latest",
        docker_client=None,
        name="",
        remove=True,
        detach=False,
        journal_tag="",
        debug=False,
    ):
        super().__init__(
            image=image,
            tag=tag,
            docker_client=docker_client,
            name=name,
            remove=remove,
            detach=detach,
            journal_tag=journal_tag,
        )
        if debug:
            self.run_cmd = "opp_run_dbg"
        else:
            self.run_cmd = "opp_run"

    def _apply_default_environment(self):
        super()._apply_default_environment()
        nedpath = (
            subprocess.check_output(f"{os.environ['ROVER_MAIN']}/scripts/nedpath")
            .decode("utf-8")
            .strip()
        )
        self.environment["NEDPATH"] = nedpath

    @staticmethod
    def __build_base_opp_run(base_cmd):
        if type(base_cmd) == str:
            cmd = [base_cmd]
        else:
            cmd = base_cmd
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

    def exec_opp_run_details(
        self,
        opp_ini="omnetpp.ini",
        config="final",
        result_dir="results",
        experiment_label="out",
        run_args_override=None,
        **kwargs,
    ):
        cmd = self.__build_base_opp_run(self.run_cmd)
        cmd.extend(["-c", config])
        if experiment_label is not None:
            cmd.extend([f"--experiment-label={experiment_label}"])
        cmd.extend([f"--result-dir={result_dir}"])
        cmd.extend(["-q", "rundetails"])
        cmd.append(opp_ini)

        return self.run(cmd, **run_args_override)

    def exec_opp_run_all(
        self,
        opp_ini="omnetpp.ini",
        config="final",
        result_dir="results",
        experiment_label="out",
        jobs=-1,
        run_args_override=None,
    ):
        cmd = ["opp_run_all"]
        if jobs > 0:
            cmd.extend(["-j", jobs])
        cmd = self.__build_base_opp_run(cmd)
        cmd.extend(["-c", config])
        if experiment_label is not None:
            cmd.extend([f"--experiment-label={experiment_label}"])
        cmd.extend([f"--result-dir={result_dir}"])
        cmd.append(opp_ini)

        return self.run(cmd, **run_args_override)

    def exec_opp_run(
        self,
        opp_ini="omnetpp.ini",
        config="final",
        result_dir="results",
        experiment_label="out",
        run_args_override=None,
        **kwargs,
    ):
        """
        Execute opp_run in container.
        """
        cmd = self.run_cmd
        cmd = self.__build_base_opp_run(cmd)
        cmd.extend(["-c", config])
        if experiment_label is not None:
            cmd.extend([f"--experiment-label={experiment_label}"])
        cmd.extend([f"--result-dir={result_dir}"])
        cmd.append(opp_ini)

        return self.run(cmd, **run_args_override)

    def set_run_args(self, run_args=None):
        super().set_run_args()
