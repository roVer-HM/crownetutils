from __future__ import annotations

import base64
import copy
import json
import os
import re
import shutil
from dbm import dumb
from email.mime import base
from glob import glob
from multiprocessing.sharedctypes import Value
from pprint import pprint
from random import seed
from time import time_ns
from typing import Dict, List, Tuple

from suqc.CommandBuilder.JarCommand import JarCommand
from suqc.request import DictVariation
from suqc.utils.SeedManager.OmnetSeedManager import OmnetSeedManager


def _parse_output(
    base_dir: str, output: str, scenario: Tuple[str, str], files: List[str]
):
    scenario, s_name = scenario
    trace_list = {}
    for f in glob(os.path.join(base_dir, output, "vadere_output/*_output")):
        print(f"parse {f}")
        with open(
            os.path.join(f, scenario.split("/")[-1]), "r", encoding="utf-8"
        ) as fd:
            scenario_json = json.load(fd)

        seed = scenario_json["scenario"]["attributesSimulation"]["fixedSeed"]
        seed_list = {}

        for file in files:
            file_base = os.path.basename(file)
            if "." in file_base:
                file_base = file_base.split(".")[0]
                file_suffix = file_base.split(".")[-1]
            else:
                raise ValueError("file name must contain suffix.")
            name = f"{file_base}_{s_name}_{seed}.{file_suffix}"
            src = os.path.join(f, file)
            dst = os.path.join(base_dir, name)
            seed_list[os.path.basename(file)] = name
            if not os.path.exists(src):
                raise ValueError(f"expected file {src} not found")

            print(f"copy: {src} --> {dst}")
            shutil.copyfile(src, dst)

        trace_list[seed] = seed_list
        with open(os.path.join(base_dir, f"trace_list_{s_name}.json"), "w") as fd:
            json.dump(trace_list, fd, sort_keys=True, indent=2)


def read_traces(base_dir: str, trace_list_only: bool = False):
    traces = glob(os.path.join(base_dir, "trace_list_*.json"), recursive=False)
    re_name = re.compile(".*trace_list_(?P<scenario_name>.*?)\.json")
    ret = {}
    for trace in traces:
        s_name = re_name.match(trace).groupdict()["scenario_name"]
        with open(trace, "r") as fd:
            trace_json = json.load(fd)
        ret[s_name] = trace_json

    if trace_list_only:
        ret_list = []
        for v in ret.values():
            for files in v.values():
                ret_list.append(files["trace.bonnMotion"])
        return ret_list
    else:
        return ret


def write_seed_paring(seed, paring, base_output_path, comment: str = "") -> None:
    vadere_seeds, opp_seeds = paring
    os.makedirs(base_output_path, exist_ok=True)
    with open(os.path.join(base_output_path, "omnetSeedManager.json"), "w") as fd:
        out = {
            "seed": seed,
            "reps": len(vadere_seeds),
            "vadere_seeds": vadere_seeds,
            "omnet_seeds": opp_seeds,
            "opp_to_vadere": {o: v for o, v in list(zip(opp_seeds, vadere_seeds))},
            "vadere_to_opp": {v: o for o, v in list(zip(opp_seeds, vadere_seeds))},
            "comment": comment,
        }
        print("write seed paring")
        pprint(out)
        json.dump(out, fd, indent=2, sort_keys=True)


def read_seeds(base_path: str, file_name: str = "omnetSeedMangaer.json") -> dict:
    with open(os.path.join(base_path, file_name), "r") as fd:
        return json.load(fd)


def generate_traces(
    scenario: str,
    scenario_name: str,
    par_var_default: dict,
    base_output_path: str,
    keep_files: List[str],
    vadere_seeds: List[int],
    cmd: JarCommand | None = None,
    jobs: int = 6,
    remove_output: bool = True,
) -> Dict[Tuple[str, str]]:
    """Create BonnMotion traces using vadere with ONE parameter set.

    Args:
        scenario (str): absolute path of scenario used in study
        scenario_name (str): new name of trace files in case the same scenario is used multiple times.
        par_var_default (dict): single parameter variation used
        base_output_path (str): output path where the suqc base output directory is created
        vadere_seeds (List[int]): list of seeds. len(vadere_seeds) is the number of repetions used.
        cmd (JarCommand | None, optional): vadere executable to use
        jobs (int, optional): number of parallel runs. Defaults to 6.
        remove_output (bool, optional): remove suqc output and only keep traces. Defaults to True.
        override_output (bool, optional): _description_. Defaults to True.

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        Dict[Tuple[str, str]]: _description_
    """

    if cmd is None:
        cmd_path = os.path.join(
            os.environ["CROWNET_HOME"],
            "vadere/VadereSimulator/target/vadere-console.jar",
        )
        if not os.path.exists(cmd_path):
            raise ValueError(f"Cannot find vadere executable at {cmd_path}")
        cmd = (
            JarCommand(jar_file=cmd_path)
            .add_option("-enableassertions")
            .main_class("suq")
        )

    par_var = []
    for seed in vadere_seeds:
        # print(f"ues seed: {seed}")
        var = copy.deepcopy(par_var_default)
        var.update(
            {
                "attributesSimulation.useFixedSeed": True,
                "attributesSimulation.fixedSeed": int(seed),
            }
        )
        par_var.append(var)

    if not os.path.isabs(scenario):
        raise ValueError(f"scenario mut be provided as an absolute path.")
    os.makedirs(base_output_path, exist_ok=True)
    out = f"{os.path.basename(scenario).split('.')[0]}.out"
    setup: DictVariation = DictVariation(
        scenario_path=scenario,
        parameter_dict_list=par_var,
        qoi="trace.bonnMotion",
        model=cmd,
        scenario_runs=1,
        post_changes=None,
        output_path=base_output_path,
        output_folder=out,
        remove_output=False,
    )
    try:
        print(f"run for scenario: {scenario}")
        setup.run(jobs)
    except Exception:
        pass

    _parse_output(base_output_path, out, (scenario, scenario_name), keep_files)
    if remove_output:
        shutil.rmtree(os.path.join(base_output_path, out))
