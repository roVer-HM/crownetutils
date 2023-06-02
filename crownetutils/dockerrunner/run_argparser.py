import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from crownetutils.dockerrunner import DockerCfg, SimulationDispatcher
from crownetutils.dockerrunner.dockerrunner import DockerCleanup, DockerReuse
from crownetutils.dockerrunner.simulators.controllerrunner import add_control_arguments
from crownetutils.dockerrunner.simulators.omnetrunner import add_omnet_arguments
from crownetutils.dockerrunner.simulators.sumorunner import add_sumo_arguments
from crownetutils.dockerrunner.simulators.vadererunner import add_vadere_arguments
from crownetutils.entrypoint.parser import SubstituteAction
from crownetutils.utils.logging import levels, logger, set_format, set_level


def result_dir_with_opp(ns, working_dir) -> str:
    """
    !Result directory callback
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
    """Used with vader only setup.
    !Result directory callback
    """
    if os.path.abspath(ns["result_dir"]):
        return ns["result_dir"]
    else:
        return os.path.join(working_dir, ns["result_dir"])


def read_sim_run_context(runner: Any, cfg_json: dict) -> Dict:
    """
    read file and search for "cmd_args" key and reload namespace from this.
    Ensure that 'config' subcommand is not present to prevent loop
    """

    if "cmd_args" not in cfg_json:
        raise argparse.ArgumentError(
            f"expected 'cmd_args' key in config file but found {cfg_json.keys()}"
        )

    cmd_args = cfg_json["cmd_args"]
    if not isinstance(cmd_args, list):
        raise argparse.ArgumentError(
            f"expected list value for 'cmd_args' but got {type(cmd_args)}"
        )

    if cmd_args[0] == "config":
        raise argparse.ArgumentError(
            f"Parse loop detected. The config file cannot contain the 'config' subcommand. [{cmd_args}]"
        )

    logger.info(f"Load config {cmd_args}")
    return parse_run_script_arguments(runner=runner, args=cmd_args)


def parse_run_script_arguments(runner: SimulationDispatcher, args=None) -> Dict:
    _args: List[str] = sys.argv[1:] if args is None else args

    # parse arguments
    main: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="BaseRunner",
        description=f"Used docker registry: {DockerCfg.registry}",
    )
    parent: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    # arguments used by all sub-commands
    _add_base_arguments(parser=parent)

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

    # omnet
    omnet_parser: argparse.ArgumentParser = sub.add_parser(
        "omnet", help="omnet subparser", parents=[parent]
    )
    add_omnet_arguments(parser=omnet_parser, args=_args)
    omnet_parser.set_defaults(main_func=runner.run_simulation_omnet)
    omnet_parser.set_defaults(result_dir_callback=result_dir_with_opp)

    # config file
    cfg_parser: argparse.ArgumentParser = sub.add_parser(
        "config",
        help="read configuration from json file",
    )
    cfg_parser.add_argument(
        "-f",
        "--file",
        dest="cfg_file",
        required=True,
        nargs=1,
        help="Any json file which contains the key 'cmd_args'. The value must be a List.",
    )

    # post processing
    post_parser: argparse.ArgumentParser = sub.add_parser(
        "post-processing",
        help="execute postprocessing on given output path",
        parents=[parent],
    )
    post_parser.set_defaults(result_dir_callback=result_dir_vadere_only)
    post_parser.set_defaults(main_func=runner.run_postprocessing_only)
    post_parser.add_argument(
        "--override-hdf",
        dest="hdf_override",
        action="store_true",
        default=False,
        required=False,
        help="If set override existing hdf files",
    )
    post_parser.add_argument(
        "--selected-only",
        dest="hdf_selected_cells_only",
        action="store_true",
        default=False,
        required=False,
        help="only parse selected measures during hdf creation",
    )

    parsed_args = main.parse_args(_args)
    ns = vars(parsed_args)
    if "cfg_file" in ns:
        cfg_file = ns["cfg_file"][0]
        with open(cfg_file, "r", encoding="utf-8") as fd:
            cfg_json = json.load(fd)
        return read_sim_run_context(runner, cfg_json)

    level_idx = ns["verbose"]
    set_level(levels[level_idx])
    set_format("%(asctime)s:%(module)s:%(levelname)s> %(message)s")

    return ns


def _add_base_arguments(parser: argparse.ArgumentParser):
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
        "--log-docker-stats",
        dest="write_container_log_stats",
        action="store_true",
        default=False,
        required=False,
        help="If true save docker stats for containers in result dir <result>/container_stats_<name>.out",
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
        type=DockerReuse,
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
