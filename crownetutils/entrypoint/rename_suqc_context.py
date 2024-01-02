import datetime
import os
import sys
from difflib import ndiff

from crownetutils.analysis.common import RunContext, SuqcStudy
from crownetutils.utils import yesno
from crownetutils.utils.logging import logger


def main_rename_suqc_context(path, ask_user, dry_run: bool, **kwargs):
    study = SuqcStudy(path)

    missing_context = []
    ctx_study_root = {}
    study_root = os.path.abspath(path)
    for run_id, run_data in study.get_run_items():
        if "run" in run_data:
            try:
                ctx: RunContext = study.get_run_context(run_id)
                # study root is the second parent from the run_context location
                if not ctx.check_study_root(study_root):
                    _ctx_root = ctx.study_root
                    _l = ctx_study_root.get(_ctx_root, [])
                    _l.append(run_id)
                    ctx_study_root[_ctx_root] = _l
            except FileNotFoundError() as e:
                missing_context.append(run_id)

    sim_count = len(study.get_run_items())
    print(f"current study root: {study_root}")
    print("-" * 80)
    for _root, runs in ctx_study_root.items():
        print(f"found study root missmatch in {len(runs)}/{sim_count} simualtions:")
        diff = ndiff([f"{_root}\n"], [f"{study_root}\n"])
        print("".join(diff), end="")
        print("-" * 80)
        print("")

    if len(ctx_study_root) == 0:
        print("Study root correct. Nothing to do.")
        sys.exit(0)
    else:
        print("Found missmaching study roots.")

    if dry_run:
        sys.exit(0)

    if ask_user:
        if not yesno.query_yes_no(question="Replace study root?", default=True):
            sys.exit(0)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
    print(f"repalce old study root with {study_root}...")
    for run_id, run_data in study.get_run_items():
        if "run" in run_data:
            try:
                print(f"process {run_data['run']}")
                ctx = study.get_run_context(run_id)
                ctx.move_study_root(
                    new_study_root=study_root,
                    bak_suffix=f"_bak{now}",
                    override_backup=False,
                )
            except FileNotFoundError() as e:
                pass

    print()

    print()
