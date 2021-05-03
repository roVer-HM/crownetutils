import os

from roveranalyzer.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)
from roveranalyzer.entrypoint.parser import ArgList
from roveranalyzer.simulators.opp.configuration import CrowNetConfig


class OppRunner(DockerRunner):
    def __init__(
        self,
        image="sam-dev.cs.hm.edu:5023/rover/crownet/omnetpp",
        tag="latest",
        docker_client=None,
        name="",
        cleanup_policy=DockerCleanup.REMOVE,
        reuse_policy=DockerReuse.REMOVE_STOPPED,
        detach=False,
        journal_tag="",
        run_cmd="CROWNET",
    ):
        super().__init__(
            image=image,
            tag=tag,
            docker_client=docker_client,
            name=name,
            cleanup_policy=cleanup_policy,
            reuse_policy=reuse_policy,
            detach=detach,
            journal_tag=journal_tag,
        )
        if len(os.path.split(run_cmd)[0]) == 0 and "CROWNET" in run_cmd:
            # run_cmd only contains executable without path. Assume default location
            self.run_cmd = CrowNetConfig.join_home(f"crownet/src/{run_cmd}")
        else:
            self.run_cmd = run_cmd

    def _apply_default_environment(self):
        super()._apply_default_environment()

    def set_run_args(self, run_args=None):
        super().set_run_args()

    @staticmethod
    def __build_base_opp_run(base_cmd):
        if type(base_cmd) == str:
            cmd = [base_cmd]
        else:
            cmd = base_cmd
        cmd.extend(["-u", "Cmdenv"])
        cmd.extend(["-l", CrowNetConfig.join_home("inet4/src/INET")])
        # cmd.extend(["-l", CrowNetConfig.join_home("crownet/src/CROWNET")])
        cmd.extend(["-l", CrowNetConfig.join_home("simulte/src/lte")])
        cmd.extend(["-l", CrowNetConfig.join_home("veins/src/veins")])
        cmd.extend(
            [
                "-l",
                CrowNetConfig.join_home("veins/subprojects/veins_inet/src/veins_inet"),
            ]
        )
        return cmd

    @staticmethod
    def create_arg_list(
        base_args: ArgList,
        result_dir,
        experiment_label,
    ):
        _arg = ArgList.from_list(base_args.data)
        _arg.add(f"--result-dir={result_dir}")
        _arg.add(f"--experiment-label={experiment_label}")
        return _arg

    def exec_opp_run(
        self,
        arg_list: ArgList,
        result_dir,
        experiment_label,
        run_args_override=None,
    ):
        """
        Execute opp_run in container.
        """
        _arg = ArgList.from_list(arg_list.data)
        _arg.add(f"--result-dir={result_dir}")
        _arg.add(f"--experiment-label={experiment_label}")
        _arg.add(self.run_cmd, pos=0)

        return self.run(_arg.to_string(), **run_args_override)
