import argparse
from typing import List

from roveranalyzer.analysis.common import SuqcRun


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
    post.add_argument("--log", action="store_true", default=False, required=False)
    post.set_defaults(main_func=lambda ns: SuqcRun.rerun_postprocessing(**vars(ns)))

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
        choices=["missing", "failed", "all"],
        default="missing",
    )
    suqcrerun.set_defaults(main_func=lambda ns: SuqcRun.rerun_simulations(**vars(ns)))
