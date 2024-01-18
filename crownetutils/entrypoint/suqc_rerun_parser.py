import argparse
from typing import List

from crownetutils.analysis.common import SuqcStudy
from crownetutils.entrypoint.rename_suqc_context import main_rename_suqc_context


def append_suqc_rerun_parser(
    sub: argparse.ArgumentParser, parents: List[argparse.ArgumentParser]
):
    post = sub.add_parser(
        "post-processing",
        help="Run post processing on selected Simulation environment",
        parents=parents,
    )

    post.add_argument(
        "--suqc-dir", dest="path", required=True, help="Suqc Simulation folder"
    )
    post.add_argument(
        "-j",
        "--jobs",
        dest="jobs",
        type=int,
        default=4,
        help="Number of parallel runs",
    )
    post.add_argument(
        "-l",
        "--list",
        dest="list_only",
        action="store_true",
        default=False,
        help="Only ist run ids of missing/failed runs",
    )
    post.add_argument(
        "--filter",
        help="List of runs. (e.g. 1,4,7-9,22-). Intervals are inclusive. Default: 'all'",
        required=False,
        default="all",
    )
    post.add_argument(
        "--what",
        "-w",
        dest="what",
        help="What should be rerun",
        choices=["failed", "all"],
        default="failed",
    )

    post.add_argument(
        "--qoi",
        action="append",
        nargs="+",
        help="Override qoi argument given in the context file of each run with these here",
        default="all",
        required=False,
        type=str,
    )

    post.add_argument(
        "--log",
        action="store_true",
        default=False,
        required=False,
        help="If flag present each subprocess will create a separate logfile with the verbosity set by -v, -vv, -vvv",
    )

    post.add_argument(
        "--verbose",
        "-v",
        dest="verbose",
        action="count",
        default=0,
        help="Set verbosity of command. From warnings and errors only (-v) to debug output (-vvv)",
    )
    post.set_defaults(main_func=lambda ns: SuqcStudy.rerun_postprocessing(**vars(ns)))

    suqcmove: argparse.ArgumentParser = sub.add_parser(
        "suqc-update-context",
        help="""A suqc Study contains absolute paths in the runContext.json to repate a 
        simulation in case it fails. When the study directory is moved reapting simualtions 
        or postprocessing steps is not possible. This tool will update the runContext.json to 
        the new location of the study directory.""",
        parents=parents,
    )

    suqcmove.add_argument(
        "--suqc-dir", dest="path", required=True, help="Suqc Simulation folder"
    )
    suqcmove.add_argument("--dry-run", required=False, action="store_true", help="Show")
    suqcmove.add_argument(
        "-y",
        "--yes",
        dest="ask_user",
        required=False,
        action="store_false",
        default=True,
        help="Yes to all user input.",
    )

    suqcmove.set_defaults(main_func=lambda ns: main_rename_suqc_context(**vars(ns)))

    suqcrerun: argparse.ArgumentParser = sub.add_parser(
        "suqc-rerun",
        help="Rerun failed or missing runs of given Study dir",
        parents=parents,
    )
    suqcrerun.add_argument(
        "--suqc-dir", dest="path", required=True, help="Suqc Simulation folder"
    )
    suqcrerun.add_argument(
        "-j",
        "--jobs",
        dest="jobs",
        type=int,
        default=4,
        help="Number of parallel runs",
    )
    suqcrerun.add_argument(
        "--what",
        "-w",
        dest="what",
        help="What should be rerun",
        choices=["failed", "all"],
        default="failed",
    )
    suqcrerun.add_argument(
        "-l",
        "--list",
        dest="list_only",
        action="store_true",
        default=False,
        help="Only ist run ids of missing/failed runs",
    )
    suqcrerun.add_argument(
        "--filter",
        help="List of runs. (e.g. 1,4,7-9,22-). Intervals are inclusive. Default: 'all'",
        required=False,
        default="all",
    )
    suqcrerun.set_defaults(main_func=lambda ns: SuqcStudy.rerun_simulations(**vars(ns)))
