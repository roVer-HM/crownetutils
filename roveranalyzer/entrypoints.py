import argparse
from typing import List

from roveranalyzer.analysis.common import SuqcStudy


def suqc_run_append_parser(
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
        "--failed-only",
        action="store_true",
        default=False,
        required=False,
        help="Tries to guess based on log file in run folders if postprocessing failed.",
    )
    post.add_argument("--log", action="store_true", default=False, required=False)
    post.set_defaults(main_func=lambda ns: SuqcStudy.rerun_postprocessing(**vars(ns)))

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
