import argparse
import os
from typing import List

from crownetutils.dockerrunner import DockerCfg
from crownetutils.dockerrunner.dockerrunner import (
    DockerCleanup,
    DockerReuse,
    DockerRunner,
)
from crownetutils.entrypoint.parser import ArgList, SimulationArgAction, filter_options
from crownetutils.utils.logging import logger
from crownetutils.utils.path import PathHelper


class OppRunner(DockerRunner):
    def __init__(
        self,
        image=DockerCfg("omnetpp"),
        tag=DockerCfg.get_default_tag(DockerCfg.VAR_OPP_TAG),
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
            self.run_cmd = PathHelper.crownet_home().join(f"crownet/src/{run_cmd}")
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

        crownet_home = PathHelper.crownet_home()
        cmd.extend(["-u", "Cmdenv"])
        cmd.extend(["-l", crownet_home.join("inet4/src/INET")])
        cmd.extend(["-l", crownet_home.join("simulte/src/lte")])
        cmd.extend(["-l", crownet_home.join("veins/src/veins")])
        cmd.extend(
            [
                "-l",
                crownet_home.join("veins/subprojects/veins_inet/src/veins_inet"),
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

        logger.debug(f"start omnett++ container(exec_opp_run)")
        logger.debug(f"cmd: {_arg.to_string()}")
        return self.run(_arg.to_string(), **run_args_override)


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
        default=DockerCfg.get_default_tag(DockerCfg.VAR_OPP_TAG),
        required=False,
        help=f"Choose Omnet container. (Default: {DockerCfg.get_default_tag(DockerCfg.VAR_OPP_TAG)})",
    )
