from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import time
from glob import glob
from os.path import basename, join
from typing import Tuple

import pandas as pd
from hjson import OrderedDict
from omnetinireader.config_parser import ObjectValue, OppConfigFileBase, OppConfigType

import roveranalyzer.simulators.crownet.dcd as Dcd
import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.entrypoint.parser import ArgList
from roveranalyzer.simulators.crownet.runner import read_config_file
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import BaseHdfProvider
from roveranalyzer.utils import Project, logger


class AnalysisBase:
    @classmethod
    def builder_from_output_folder(
        cls,
        data_root: str,
        hdf_file="data.h5",
        vec_name="vars_rep_0.vec",
        sca_name="vars_rep_0.sca",
        network_name="World",
        epsg_base=Project.UTM_32N,
    ) -> Tuple[str, Dcd.DcdHdfBuilder, OMNeT.CrownetSql]:

        builder = Dcd.DcdHdfBuilder.get(hdf_file, data_root).epsg(epsg_base)

        sql = OMNeT.CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
        return data_root, builder, sql

    @classmethod
    def builder_from_suqc_output(
        cls,
        data_root: str,
        out,
        parameter_id,
        run_id=0,
        hdf_file="data.h5",
        vec_name="vars_rep_0.vec",
        sca_name="vars_rep_0.sca",
        network_name="World",
        epsg_base=Project.UTM_32N,
    ) -> Tuple[str, Dcd.DcdHdfBuilder, OMNeT.CrownetSql]:

        data_root = join(
            data_root, "simulation_runs/outputs", f"Sample_{parameter_id}_{run_id}", out
        )
        builder = Dcd.DcdHdfBuilder.get(hdf_file, data_root).epsg(epsg_base)

        sql = OMNeT.CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
        return data_root, builder, sql

    @staticmethod
    def find_selection_method(builder: Dcd.DcdHdfBuilder):
        p = builder.build().map_p
        return p.get_attribute("used_selection")


class RunContext:
    class _dummy_runner:
        def run_simulation_omnet(self, *args, **kwargs):
            raise NotImplementedError

        def run_simulation_omnet_sumo(self, *args, **kwargs):
            raise NotImplementedError

        def run_vadere(self, *args, **kwargs):
            raise NotImplementedError

        def run_simulation_vadere_ctl(self, *args, **kwargs):
            raise NotImplementedError

        def run_simulation_omnet_vadere(self, *args, **kwargs):
            raise NotImplementedError

        def run_simulation_vadere_omnet_ctl(self, *args, **kwargs):
            raise NotImplementedError

        def run_post_only(self, *args, **kwargs):
            raise NotImplementedError

    @classmethod
    def from_path(cls, path):
        with open(path, "r", encoding="utf-8") as fd:
            return cls(json.load(fd))

    def __init__(self, data) -> None:
        self.data = data
        self._ns = read_config_file(self._dummy_runner(), self.data)
        self.args: ArgList = ArgList.from_flat_list(self.data["cmd_args"])

    @property
    def cwd(self):
        return self.data["cwd"]

    @property
    def oppini_name(self):
        return self.args.get_value("--opp.-f", default="omnetpp.ini")

    @property
    def oppini_path(self):
        return join(self.cwd, self.oppini_name)

    @property
    def oppini(self) -> OppConfigFileBase:
        config = self.args.get_value("--opp.-c", "final")
        return OppConfigFileBase.from_path(
            self.oppini_path, config=config, cfg_type=OppConfigType.READ_ONLY
        )

    @property
    def resultdir(self):
        return self._ns["result_dir_callback"](self._ns, self._ns["result_dir"])

    @property
    def sample_name(self):
        sample = self.args.get_value("--resultdir")
        return sample.split(os.sep)[-1]

    def create_postprocessing_args(self, qoi_default="all"):
        return {
            "cwd": self.cwd,
            "script_name": self.data.get("script", "run_script.py"),
            "args": [
                "post-processing",
                "--qoi",
                self.args.get_value("--qoi", qoi_default),
                "--resultdir",
                self.resultdir,
            ],
        }

    @staticmethod
    def exec_runscript(args: dict, out=subprocess.DEVNULL, err=subprocess.DEVNULL):

        cmd = [os.path.join(args["cwd"], args["script_name"]), *args["args"]]
        if args["log"]:
            fd = open(os.path.join(args["cwd"], "log.out"), "w")
            out = fd
            err = fd
        try:
            return_code: int = subprocess.check_call(
                cmd,
                env=os.environ,
                stdout=out,
                stderr=err,
                cwd=args["cwd"],
            )
        except Exception as e:
            print(e)
            print(f"Simulation failed: {cmd}")
            return_code = -1
        finally:
            if args["log"]:
                fd.close()

        return return_code


class Simulation:
    """Access output of one simulation, accessing different types of output generated
    such as scalar and vector files as well as density maps, vadere or sumo output.

    """

    @classmethod
    def from_context(cls, ctx: RunContext, label=""):
        return cls(ctx.resultdir, label=label, run_context=ctx)

    @classmethod
    def from_suqc_result(cls, data_root, label=""):
        for i, p in enumerate(data_root.split(os.sep)[::-1]):
            if p.startswith("Sample"):
                label = f"{p}_{label}"
                runcontext = join(data_root, "../../../", p, "runContext.json")
                runcontext = os.path.abspath(runcontext.replace("Sample_", "Sample__"))
                o = cls(data_root, label, RunContext.from_path(runcontext))
                o.run_context = RunContext.from_path(runcontext)
                return o
        raise ValueError("data_root not an suq-controller output directory")

    def __init__(self, data_root, label, run_context: RunContext | None = None):
        self.label = label
        (
            self.data_root,
            self.builder,
            self.sql,
        ) = AnalysisBase.builder_from_output_folder(data_root)
        self.pos: BaseHdfProvider = BaseHdfProvider(
            join(self.data_root, "trajectories.h5"), group="trajectories"
        )
        self.run_context: RunContext = run_context

    def get_base_provider(self, group_name, path=None) -> BaseHdfProvider:
        return BaseHdfProvider(
            hdf_path=path or join(self.data_root, ""), group=group_name
        )

    def get_dcdMap(self):
        sel = self.builder.map_p.get_attribute("used_selection")
        if sel is None:
            raise ValueError("selection not set!")
        return self.builder.build_dcdMap(selection=list(sel)[0])

    def get_run_description_0001(self):
        cfg = self.run_context.oppini
        map: ObjectValue = cfg["*.pNode[*].app[1].app.mapCfg"]
        ret = []
        ret.append(["Sample", self.run_context.sample_name])
        if map.name == "MapCfgYmfPlusDistStep":
            ret.append(
                [
                    "mapCfg",
                    f"a: {map['alpha']}, stepDist: {map['stepDist']}|{map['zeroStep']}",
                ]
            )
        elif map.name == "YmfPlusDistStep":
            ret.append(["mapCfg", f"alpha: {map['alpha']}"])
        else:
            ret.append(["mapCfg", f"{map.name}"])
        ret.append(["seed", cfg.get("seed-set")])
        ret.append(["sim-time", cfg.get("sim-time-limit")])

        ret.append(
            [
                "Beacon-interval",
                cfg.get("*.pNode[*].app[0].scheduler.generationInterval"),
            ]
        )
        ret.append(
            ["Map-interval", cfg.get("*.pNode[*].app[1].scheduler.generationInterval")]
        )
        ret.append(
            ["Map-data/interval", cfg.get("*.pNode[*].app[1].scheduler.amoutOfData")]
        )
        ret = pd.DataFrame(ret).T
        ret.columns = ret.loc[0]
        ret = ret.drop(0)
        ret.set_index(["Sample"])
        return ret

    @staticmethod
    def asset_pdf_path(name, base="assets/pdf", suffix=""):
        base_name = basename(name).replace(".pdf", "")
        return join(base, f"{base_name}{suffix}.pdf")

    @classmethod
    def copy_pdf(cls, name, sim: Simulation, base):
        shutil.copyfile(
            src=join(sim.data_root, name),
            dst=cls.asset_pdf_path(name, base=base, suffix=sim.label),
        )


class SuqcRun:
    """Class representing a Suq-Controller environment containing the definition and
    results for one simulation study.


    The class tries to guess the simulation prefix chosen during the study execution.
    It then links the each run (config files for *one* parameter variation) with the
    corresponding outputs.

    For easy access to a single run outputs see Simulation class

    Returns:
        _type_: _description_
    """

    run_pattern = re.compile(r"(^.*?)_+(\d+)_(\d+)$")

    @classmethod
    def guess_run_prefix(cls, base_path):
        logger.info("no run_prefix supplied. Try to guess prefix from run folder")
        runs = glob(
            join(
                base_path,
                "simulation_runs/*",
            )
        )
        runs = [os.path.basename(p) for p in runs if not p.endswith("outputs")]
        run_prefix = set()
        for run in runs:
            if match := cls.run_pattern.match(run):
                run_prefix.add(match.group(1))

        if len(run_prefix) == 1:
            return run_prefix.pop()
        else:
            raise ValueError(f"None or more than one run_prefix fond '{run_prefix}'")

    @classmethod
    def get_run_paths(cls, base_path, run_prefix):
        runs = glob(join(base_path, "simulation_runs/*"))
        runs = [r for r in runs if not r.endswith("outputs")]
        outputs = glob(
            join(base_path, f"simulation_runs/outputs/{run_prefix}*/*"), recursive=False
        )
        outputs = [o for o in outputs if os.path.isdir(o)]

        ret = {}
        p_run = re.compile(f"^.*{os.sep}{run_prefix}_+(\\d+)_(\\d+)$")
        p_out = re.compile(f"^.*{os.sep}{run_prefix}_+(\\d+)_(\\d+){os.sep}.*$")

        for r in runs:
            if match := p_run.match(r):
                key = (int(match.group(1)), int(match.group(2)))
                e = ret.get(key, {})
                e["run"] = r
                ret[key] = e
        for o in outputs:
            if match := p_out.match(o):
                key = (int(match.group(1)), int(match.group(2)))
                e = ret.get(key, {})
                e["out"] = o
                ret[key] = e
        for key, val in ret.items():
            if not all(i in val for i in ["run", "out"]):
                logger.warning(f"For run {key} one or more directories are missing.")

        return OrderedDict(sorted(ret.items(), key=lambda i: (i[0][0], i[0][1])))

    def __init__(self, base_path, run_prefix="") -> None:
        self.base_path = base_path
        self.name = basename(base_path)
        if run_prefix == "":
            self.run_prefix = self.guess_run_prefix(base_path)
        else:
            self.run_prefix = run_prefix
        self.runs = self.get_run_paths(base_path, self.run_prefix)

    @property
    def run_paths(self):
        return [r["run"] for r in self.runs]

    @property
    def out_paths(self):
        return [r["out"] for r in self.runs]

    @property
    def output_folder(self):
        return join(self.base_path, "simulation_runs", "outputs")

    def get_path(self, run, path_type):
        return self.runs[run][path_type]

    def run_path(self, key):
        return self.get_path(key, "run")

    def out_path(self, key):
        return self.get_path(key, "out")

    def get_sim(self, key) -> Simulation:
        if isinstance(key, int):
            key = (key, 0)
        return self.get_run_as_sim(key)

    def get_run_as_sim(self, key):
        run = self.runs[key]
        ctx = RunContext.from_path(join(run["run"], "runContext.json"))
        lbl = f"{self.name}_{self.run_prefix}_{key[0]}_{key[1]}"
        return Simulation.from_context(ctx, lbl)

    def get_simulations(self):
        return [self.get_run_as_sim(k) for k in self.runs.keys()]

    def get_simulation_dict(self, lbl_key=False):

        ret = {k: self.get_run_as_sim(k) for k in self.runs.keys()}

        if lbl_key:
            ret = {v.label: v for _, v in ret.items()}

        return OrderedDict(sorted(ret.items(), key=lambda i: (i[0][0], i[0][1])))

    @classmethod
    def rerun_postprocessing(cls, path: str, njobs=4, log=False):
        run: SuqcRun = cls(path)

        args = []
        for sim in run.get_simulations():
            _arg = sim.run_context.create_postprocessing_args()
            _arg["log"] = log
            args.append(_arg)

        with multiprocessing.Pool(processes=njobs) as pool:
            ret = pool.map(func=RunContext.exec_runscript, iterable=args)

        return all(ret)

    @staticmethod
    def main_postprocessing(ns: argparse.ArgumentParser):
        ret = SuqcRun.rerun_postprocessing(ns.path, ns.jobs, ns.log)
        if ret:
            sys.exit(0)
        else:
            sys.exit(-1)

    @staticmethod
    def create_parser(sub: argparse.ArgumentParser):
        sub.add_argument(
            "--suqc-dir", dest="path", required=True, help="Suqc Simulation folder"
        )
        sub.add_argument(
            "-j",
            "--jobs",
            dest="jobs",
            type=int,
            default=4,
            help="Number of parallel runs",
        )
        sub.add_argument("--log", action="store_true", default=False, required=False)
        sub.set_defaults(main_func=SuqcRun.main_postprocessing)
