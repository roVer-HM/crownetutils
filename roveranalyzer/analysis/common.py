from __future__ import annotations

import datetime
import enum
import json
import os
import re
import shutil
import subprocess
import timeit as it
from ast import Param
from contextlib import contextmanager
from functools import partial
from glob import glob
from multiprocessing import get_context
from os.path import basename, join
from tokenize import group
from typing import (
    IO,
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterable,
    Iterator,
    List,
    Protocol,
    TextIO,
    Tuple,
)

import numpy as np
import pandas as pd
from hjson import OrderedDict
from matplotlib.backends.backend_pdf import PdfPages
from omnetinireader.config_parser import ObjectValue, OppConfigFileBase, OppConfigType

import roveranalyzer.simulators.crownet.dcd as Dcd
from roveranalyzer.analysis.base import AnalysisBase
from roveranalyzer.entrypoint.parser import ArgList
from roveranalyzer.simulators.crownet.dcd.dcd_map import DcdMap2D
from roveranalyzer.simulators.crownet.runner import read_config_file
from roveranalyzer.simulators.opp import CrownetSql
from roveranalyzer.simulators.opp.provider.hdf.IHdfProvider import BaseHdfProvider
from roveranalyzer.utils import Project, logger
from roveranalyzer.utils.misc import apply_str_filter
from roveranalyzer.utils.parallel import run_args_map, run_kwargs_map


class RunMapCreateFunction(Protocol):
    """Interface to create RunMap using output_path as the RunMap root directory.
    Use *args and **kwds for specific implementation
    """

    def __call__(self, output_path, *args: Any, **kwds: Any) -> RunMap:
        ...


class SimGroupAppendStrategy(enum.Enum):
    APPEND = 1
    DENY_APPEND = 2
    DROP_OLD = 3
    DROP_NEW = 4
    DENY_NEW = 5


class SimulationGroup:
    """A named group of Simulation objects.

    The grouped simulation have some similar property, mostly they are the of the
    parameter variation but with different seeds. The Simulation object do not have to
    be from the same SuqcStudy. If Simulation objects from different runs are combined the
    user must ensure that there are no id overlaps. Use the id_offset in the Simulation object.
    See RunMap   and SuqcStudy.update_run_map for details.
    """

    def __init__(
        self, group_name: str, data: List[Simulation], attr: dict | None = None, **kwds
    ) -> None:
        self.group_name: str = group_name
        self.simulations: List[Simulation] = data
        self.attr: dict = {} if attr is None else attr

    @property
    def lbl(self):
        """Alias for self.group_name"""
        return self.label

    @property
    def reps(self):
        """Alias for self.ids()"""
        return self.ids()

    @property
    def label(self):
        """Label of default to self.group_name if not set"""
        if "lbl" in self.attr:
            return self.attr["lbl"]
        else:
            return self.group_name

    @label.setter
    def label(self, val):
        self.attr["lbl"] = val

    def extend(self, sim_group: SimulationGroup):
        if self.group_name != sim_group.group_name:
            raise ValueError(
                f"cannot extend SimulationGroup with different names {self.group_name}!={sim_group.group_name}"
            )
        id_set = [*self.ids(), *sim_group.ids()]
        if len(id_set) != len(set(id_set)):
            raise ValueError(f"duplicated simulation ids in group found. {id_set}")
        self.simulations.extend(sim_group.simulations)

    def ids(self) -> List[int]:
        return [sim.global_id() for sim in self.simulations]

    def opp_seeds(self) -> list[int]:
        """Return OMNeT specific seed"""
        return [sim.run_context.opp_seed for sim in self.simulations]

    def mobility_seeds(self) -> list[int]:
        """Return mobility specific seed"""
        return [sim.run_context.mobility_seed for sim in self.simulations]

    def simulation_iter(self, enum: bool = False) -> Iterator[Tuple[int, Simulation]]:
        """Get iterator of all items (global_sim_id, simulation) in this group. If enum is
        set the iterator returns (run_id, global_sim_id, simulation) instead.

        Args:
            enum (bool, optional): Add global id to iterator. Defaults to False.

        Yields:
            Iterator[Tuple[int, Simulation]]: _description_
        """
        for idx, sim in enumerate(self.simulations):
            if enum:
                yield (idx, sim.global_id(), sim)
            else:
                yield (sim.global_id(), sim)

    def __getitem__(self, key) -> Simulation:
        return self.simulations[key]

    def __iter__(self) -> Iterator[Simulation]:
        return iter(self.simulations)

    def __len__(self):
        return len(self.simulations)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {hex(id(self))} group_name: {self.group_name} >"

    # def __str__(self) -> str:
    #     return self.__repr__()


class EmptySimGroupFilter:
    def __call__(self, sim_group: SimulationGroup) -> bool:
        return True


class SimGroupFilter(Protocol):
    """Callable which takes a"""

    EMPTY = EmptySimGroupFilter()

    def __call__(self, sim_group: SimulationGroup) -> bool:
        ...


class RunMap(dict):
    """Dictionary like class with label:str -> SimulationGroup

    This class allows simple handling of multiple parameter variations with multiple seeds
    per variation. RunMap inherits from dict and contains helper methods to filter and iterate
    over SimulationGroups. The RunMap is mostly used in conjunction with roveranaluyzer.utis.parallel
    to dispatch each parameter variation (i.e. SimulationGroup) to one subprocess to speed up the analysis.
    """

    def __init__(self, output_dir: str):
        self.output_dir: str = output_dir
        self._mobility_seeds = None
        os.makedirs(self.output_dir, exist_ok=True)

    @staticmethod
    def load_or_create(
        create_f: RunMapCreateFunction,
        output_path: str,
        file_name: str = "run_map.json",
        load_if_present: bool = True,
        *args,
        **kwargs,
    ) -> RunMap:
        """Load from filesystem if RunMap json exists, otherwise use provided factory function"""
        if load_if_present and os.path.exists(os.path.join(output_path, file_name)):
            return RunMap.load_from_json(os.path.join(output_path, file_name))

        run_map = create_f(output_path, *args, **kwargs)
        run_map.save_json(os.path.join(output_path, file_name))
        return run_map

    def path(self, *args):
        """Return path relative to RunMap ouput_dir"""
        return os.path.join(self.output_dir, *args)

    def path_exists(self, *args) -> bool:
        """Check if path relative to RunMap output_dir exists."""
        return os.path.exists(self.path(*args))

    def any(self, filter_f: SimGroupFilter) -> SimulationGroup:
        """Return first :class:`SimulationGroup` that adheres to filter function.

        Args:
            filter_f (SimGroupFilter): _description_

        Returns:
            SimulationGroup:
        """
        for g in self.values():
            if filter_f(g):
                return g

    def all(self, filter_f: SimGroupFilter) -> List[SimulationGroup]:
        """Return all :class:`SimulationGroup` objects that adhere to filter_f.

        Args:
            func (SimGroupFilter): Some predicate function taking a SimulationGroup object

        Returns:
            List[SimulationGroup]:
        """
        return [g for g in self.values() if filter_f(g)]

    def get_mobility_seed_set(self):
        if self._mobility_seeds is None:
            _seeds = set()
            for sg in self.values():
                sim: Simulation
                for _, sim in sg.simulation_iter():
                    _seeds.add(sim.run_context.mobility_seed)
            self._mobility_seeds = list(_seeds)
        return self._mobility_seeds

    @contextmanager
    def pdf_page(
        self, *args, keep_empty: bool = True, metadata=None
    ) -> ContextManager[PdfPages]:
        with PdfPages(self.path(*args), keep_empty=True, metadata=metadata) as pdf:
            yield pdf

    # def open(self, path, mode):

    def attr(self, group, key, _default: Any = None):
        if _default is None:
            return self[group].attr[key]
        else:
            return self[group].attr.get(key, _default)

    def get_simulation_group(
        self, sort_by: None | Callable[[SimulationGroup], Any] = None
    ) -> List[SimulationGroup]:
        if sort_by is None:
            return list(self.values())
        else:
            _g = list(self.values())
            _g.sort(key=sort_by)
            return _g

    def filtered_parameter_variations(
        self, filter_f: Callable[[str], bool] = lambda x: True
    ):
        return [v for v in self.get_simulation_group() if filter_f(v.group_name)]

    def iter(self, filter_f: Callable[[SimulationGroup], bool]):
        for g_name, g in self.items():
            if filter_f(g):
                yield g_name, g

    def append_or_add(
        self,
        sim_group: SimulationGroup,
        strategy: SimGroupAppendStrategy = SimGroupAppendStrategy.APPEND,
    ):
        if sim_group.group_name in self:
            if strategy == SimGroupAppendStrategy.APPEND:
                self[sim_group.group_name].extend(sim_group)
                self[sim_group.group_name].attr.update(sim_group.attr)
            elif strategy == SimGroupAppendStrategy.DROP_OLD:
                self[sim_group.group_name] = sim_group
                self[sim_group.group_name].attr = sim_group.attr
            elif strategy == SimGroupAppendStrategy.DROP_NEW:
                pass  # do nothing
            else:
                raise (
                    f"Cannot append/override group: {sim_group.group_name} with strategy {strategy.name}"
                )
        else:
            if strategy == SimGroupAppendStrategy.DENY_NEW:
                raise (
                    f"Cannot add new group {sim_group.group_name} with strategy {strategy.name}"
                )
            self[sim_group.group_name] = sim_group

    @property
    def max_id(self) -> int:
        """Return biggest id used in any simulation in any SimulationGroup"""
        if len(self.get_simulation_group()) == 0:
            return -1
        else:
            return max([max(sim.ids()) for sim in self.get_simulation_group()])

    def save_json(self, fd: str | TextIO | None = None):
        ret = {}
        ret["output_dir"] = self.output_dir
        ret["groups"] = {}

        for group in self.get_simulation_group():
            g = {}
            g["attr"] = group.attr
            g["group_name"] = group.group_name
            g["data"] = [
                (sim.run_context.ctx_path, sim.label, sim._id_offset) for sim in group
            ]
            ret["groups"][group.group_name] = g

        fd = self.path("run_map.json") if fd is None else fd
        if isinstance(fd, str):
            with open(fd, "w") as fd:
                json.dump(ret, fd, indent=2)
        else:
            json.dump(ret, fd, indent=2)

    @classmethod
    def load_from_json(cls, fd: str | TextIO) -> RunMap:
        if isinstance(fd, str):
            with open(fd, "r") as fd:
                map: dict = json.load(fd)
        else:
            map: dict = json.load(fd)

        ret = cls(map["output_dir"])
        for g_name, group in map["groups"].items():
            sims = [
                Simulation.from_context(ctx, label, id_offset)
                for ctx, label, id_offset in group["data"]
            ]
            sim_group = SimulationGroup(g_name, data=sims, attr=group["attr"])
            ret.append_or_add(sim_group)
        return ret

    def attr_df(self):
        """Create DataFrame with group key as index and attributes as columns"""
        idx = pd.Index(self.keys(), name="sim")
        records = [g.attr for g in self.values()]
        _df = pd.DataFrame.from_records(records, index=idx)
        _df = _df.apply(partial(pd.to_numeric, errors="ignore"))
        return _df

    def get_sim_by_id(self, glb_id) -> Simulation:
        g: SimulationGroup
        for g in self.values():
            for sim in g.simulations:
                if sim.global_id() == glb_id:
                    return sim
        raise KeyError(f"No simulation with id {glb_id} found.")

    def get_group_by_sim_id(self, glb_id) -> SimulationGroup:
        g: SimulationGroup
        for g in self.values():
            if glb_id in g.ids():
                return g
        raise KeyError(f"No simulation with id {glb_id} found.")

    def id_to_label_series(
        self,
        lbl_f: None | Callable[[Simulation], str] = None,
        enumerate_run: bool = False,
    ) -> pd.DataFrame:
        """Create a DataFrame with mapping between run_id and some label string.
        This function uses the first run_id in Parameter_Variation.reps  if lbl_f is set.

        Args:
            lbl_f (None | Callable[[int | Tuple[int, int]], str], optional): Function that takes a run_id to create a label. Defaults to None.

        Returns:
            pd.DataFrame: columns: [run_id, label]
        """
        df = []
        for idx, item in enumerate(self.items()):
            lbl: str = item[0]
            group: SimulationGroup = item[1]
            if enumerate_run:
                _df = pd.DataFrame(
                    np.array([group.ids(), group.opp_seeds(), np.arange(len(group))]).T,
                    columns=["run_id", "seed", "run_index"],
                )
            else:
                _df = pd.DataFrame(group.ids(), columns=["run_id"])
            _df["label"] = group.group_name if lbl_f is None else lbl_f(group[0])
            _df["group_index"] = idx
            df.append(_df)
        df = pd.concat(df)
        df = df.set_index(["run_id"])
        return df

    def __getitem__(self, __key) -> SimulationGroup:
        if isinstance(__key, int):
            _items = list(self.values())
            return _items[__key]
        else:
            return super().__getitem__(__key)


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
            return cls(json.load(fd), path)

    def __init__(self, data, ctx_path: str | None = None) -> None:
        self.data = data
        self.ctx_path = ctx_path
        self._ns = read_config_file(self._dummy_runner(), self.data)
        self.args: ArgList = ArgList.from_flat_list(self.data["cmd_args"])
        self._opp_seed = None
        self._m_seed = None
        self._mobility_type = None

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

    def ini_get(
        self,
        key: str,
        regex: str | None = None,
        apply: Callable[[Any], Any] = lambda x: x,
    ):
        """Return ini entry for key. Apply regex and any given `apply` function to returned value.

        Args:
            key (str): Ini-File key.
            regex (str | None, optional): Regex with _ONE_ group to apply to raw value returned. Defaults to None.
            apply (_type_, optional): Any consumer of the the returned value for further cleanup. Defaults to no action.

        Raises:
            ValueError: ValueError if regex does not return a match

        Returns:
            _type_: Ini Value
        """
        value = self.oppini[key]
        if regex is not None:
            pattern = re.compile(regex)
            match = pattern.match(value)
            if not match:
                raise ValueError(f"no match for {key}={value} in regex {pattern}")
            value = match.groups()[0]
        return apply(value)

    def ini_get_or_default(
        self,
        key: str,
        regex: str | None = None,
        apply: Callable[[Any], Any] = lambda x: x,
        default: Any = None,
    ):
        """Return ini entry for key or default. Apply regex and any given `apply` function to returned value.

        Args:
            key (str): Ini-File key.
            regex (str | None, optional): Regex with _ONE_ group to apply to raw value returned. Defaults to None.
            apply (_type_, optional): Any consumer of the the returned value for further cleanup. Defaults to no action.
            default (Any): Default value in case key does not exist.

        Returns:
            _type_: Ini Value or default.
        """

        try:
            ret = self.ini_get(key, regex, apply)
        except ValueError:
            ret = default
        return ret

    @property
    def opp_seed(self) -> int:
        """Return OMNeT (communication) seed used in simulation"""
        if self._opp_seed is None:
            self._opp_seed = int(self.oppini["seed-set"])
        return self._opp_seed

    @property
    def mobility_seed(self) -> int:
        """Return mobility seed used in simulation. This might be equal to
        the opp_seed if OMNeT internal mobility patters are used. In case of
        Vadere or trace based mobility the seed might differ."""
        if self._m_seed is None:
            _t = self.mobility_type
            if _t == "bonnMotion":
                return self._m_seed
            elif _t == "opp":
                return self._m_seed
            else:
                raise NotImplementedError("Not implemented for Vadere")
        return self._m_seed

    @property
    def mobility_type(self) -> str:
        if self._mobility_type is None:
            # check bonnMotion
            try:
                seed = self.ini_get(
                    "*.bonnMotionServer.traceFile",
                    regex=r".*_(\d+)\.bonnMotion",
                    apply=int,
                )
                self._mobility_type = "bonnMotion"
                if self._m_seed is None:
                    self._m_seed = seed
                return self._mobility_type
            except:
                pass
            # check vadere
            try:
                v = self.ini_get("**.vadereScenarioPath")
                self._mobility_type = "vadere"
                return self._mobility_type
            except:
                pass
            # assume omnet
            self._mobility_type = "opp"
            if self._m_seed is None:
                self._m_seed = self.opp_seed

        return self._mobility_type

    @property
    def par_id(self) -> int:
        return self.data["request_item"]["parameter_id"]

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

    def create_run_config_args(self):
        return {
            "cwd": self.cwd,
            "script_name": self.data.get("script", "run_script.py"),
            "args": ("config", "-f", "runContext.json"),
        }

    @staticmethod
    def exec_runscript(args: dict, out=subprocess.DEVNULL, err=subprocess.DEVNULL):

        cmd = [os.path.join(args["cwd"], args["script_name"]), *args["args"]]
        print(f"run command:\n\t\t{cmd}")
        if args["log"]:
            os.makedirs(os.path.dirname(args["cwd"]), exist_ok=True)
            fd = open(os.path.join(args["cwd"], "log.out"), "w")
            out = fd
            err = fd
        if "clean_dir" in args:
            if os.path.exists(args["clean_dir"]):
                shutil.rmtree(args["clean_dir"])
        try:
            return_code: int = subprocess.check_call(
                cmd,
                env=os.environ,
                stdout=out,
                stderr=err,
                cwd=args["cwd"],
            )
            print(f"check_return: {return_code} | {cmd}")
        except Exception as e:
            print(e)
            print(f"Simulation failed: {cmd}")
            return_code = -1
        finally:
            if args["log"]:
                fd.close()
        print(f"done: {cmd}")
        return return_code


class SimulationBase:
    def __init__(
        self, data_root: str, label: str, run_context: RunContext | None = None
    ) -> None:
        self.data_root = data_root
        self.label = label
        self.run_context: RunContext = run_context


class Simulation:
    """Builder class allows access output of *one* simulation for accessing
    different types of output generated such as scalar and vector files
    as well as density maps, vadere or sumo output.

    self.sql            CrownetSql object to access OMNeT++ output (sca, vec)
    self.get_dcdMap()   Access to Density Map related analysis
    self.run_context    Access to simulation config used during simulation
    """

    @classmethod
    def from_context(cls, ctx: str | RunContext, label="", id_offset: int = 0):
        if isinstance(ctx, str):
            ctx = RunContext.from_path(ctx)
        return cls(ctx.resultdir, label=label, run_context=ctx, id_offset=id_offset)

    @classmethod
    def from_suqc_result(cls, data_root, label="", id_offset: int = 0):
        for i, p in enumerate(data_root.split(os.sep)[::-1]):
            if p.startswith("Sample"):
                label = f"{p}_{label}"
                runcontext = join(data_root, "../../../", p, "runContext.json")
                runcontext = os.path.abspath(runcontext.replace("Sample_", "Sample__"))
                o = cls(data_root, label, RunContext.from_path(runcontext), id_offset)
                # o.run_context = RunContext.from_path(runcontext)
                return o
        raise ValueError("data_root not an suq-controller output directory")

    @classmethod
    def from_output_dir(cls, data_root, **kwds):
        data_root, builder, sql = AnalysisBase.builder_from_output_folder(
            data_root, **kwds
        )
        lbl = os.path.basename(data_root)
        c = cls(data_root, lbl, run_context=None, id_offset=0)
        c._builder = builder
        c._sql = sql
        return c

    def __init__(
        self, data_root, label, run_context: RunContext = None, id_offset: int = 0
    ):
        self.label = label
        self._builder = None
        self._sql = None
        self.data_root = data_root
        self.run_context: RunContext = run_context
        self._id_offset = id_offset

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} object at {hex(id(self))} {self.label} ({self.study_id()}[{self.global_id()}])>"

    @property
    def builder(self) -> Dcd.DcdHdfBuilder:
        if self._builder is None:
            self._builder = Dcd.DcdHdfBuilder.get("data.h5", self.data_root).epsg(
                Project.UTM_32N
            )
        return self._builder

    @property
    def sql(self) -> CrownetSql:
        if self._sql is None:
            self._sql = CrownetSql(
                vec_path=f"{self.data_root}/vars_rep_0.vec",
                sca_path=f"{self.data_root}/vars_rep_0.sca",
                network="World",
            )
        return self._sql

    def get_base_provider(self, group_name="root", path="data.h5") -> BaseHdfProvider:

        if not os.path.isabs(path):
            path = self.path(path)

        return BaseHdfProvider(hdf_path=path, group=group_name)

    def base_hdf(self, group_name) -> BaseHdfProvider:
        return self.get_base_provider(group_name, join(self.data_root, "data.h5"))

    def from_hdf(self, hdf_path: str, group: str, **kwargs) -> pd.DataFrame:
        """Extract dataframe from provided hdf file. Use kwargs to select portion of data.
        This method accepts all keywords from pandas.HDFStore.select (where, start, stop, columns)

        Args:
            hdf_path (str): Path to hdf file. If relative use simulation root as as basis.
            group (str): group to select from

        Returns:
            pd.DataFrame:
        """
        if not os.path.isabs(hdf_path):
            hdf_path = self.path(hdf_path)
        _hdf = BaseHdfProvider(hdf_path, group)
        if len(kwargs) == 0:
            return _hdf.get_dataframe(group)
        else:
            with _hdf.ctx(mode="r") as ctx:
                return ctx.select(key=group, **kwargs)

    def global_id(self):
        return self.run_context.par_id + self._id_offset

    def study_id(self):
        return self.run_context.par_id

    def path(self, *args):
        """Create path relative to Simulation object data_root directory"""
        return os.path.join(self.data_root, *args)

    @property
    def pos(self) -> BaseHdfProvider:
        return BaseHdfProvider(
            join(self.data_root, "trajectories.h5"), group="trajectories"
        )

    def get_dcdMap(self) -> DcdMap2D:
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
        try:
            shutil.copyfile(
                src=join(sim.data_root, name),
                dst=cls.asset_pdf_path(name, base=base, suffix=sim.label),
            )
        except Exception as e:
            logger.info(f"problem copying {join(sim.data_root, name)}: {e}")


class SimulationGroupFactory(Protocol):
    """Create a SimulationGroup using one simulation to access config to derive
    name or label information. **kwds must provide all necessary attributes to
    build the SimulationGroup object. Implementer might override **kwds if
    needed.
    """

    def __call__(self, sim: Simulation, **kwds: Any) -> SimulationGroup:
        ...


class NamedSimulationGroupFactory(SimulationGroupFactory):
    """SimulationGroup factory which provides group names based on name list or
    generic group names "group_{idx}" no list is provided."""

    def __init__(self, name_list: list[str] | None = None) -> None:
        self.group_count = 0
        self.name_list = name_list

    def __call__(self, sim: Simulation, **kwds: Any) -> SimulationGroup:
        if self.name_list is None:
            g_name = f"group_{self.group_count}"
        else:
            if self.group_count < len(self.name_list):
                g_name = self.name_list[self.group_count]
            else:
                raise IndexError(
                    f"Only {len(self.name_list)} group names defined but {self.group_count +1} groups requested."
                )
        self.group_count += 1
        return SimulationGroup(g_name, **kwds)


class SuqcStudy:
    """A light weight class representing a Suq-Controller environment containing the definition and
    results for one simulation study.

    The class tries to guess the simulation prefix chosen during the study execution.
    It then links the each run (config files for *one* parameter variation) with the
    corresponding outputs.

    For easy access to a single simulation (or run) outputs see Simulation class. This class only
    manages paths and does create Simulation objects lazily on demand
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

    def get_failed_missing_runs(self, run_item_filter=lambda x: True):
        def check_fail(**kwargs) -> bool:
            if "out" not in kwargs:
                return True  # failed (i.e. not started) Simulation
            out = kwargs["out"]
            opp_out = os.path.join(out, "container_opp.out")
            if not os.path.exists(opp_out):
                return True  # failed (i.e. not run completely) simulation
            with open(opp_out, "r", encoding="utf-8") as fd:
                content = fd.readlines()
                for line in content:
                    if line.startswith("<!> Error:"):
                        logger.info(f"error in {out}: {line}")
                        return (
                            True  # failed (i.e some error found in output) simulation
                        )
            journal_files = glob(os.path.join(out, "*-journal"))
            if len(journal_files) > 0:
                return True

            return False

        _items = self.get_run_items(filter=run_item_filter)
        runs_failed: List[RunContext] = [
            self.get_run_context(k) for k, v in _items if check_fail(**v)
        ]
        return runs_failed

    def __init__(self, base_path, run_prefix="") -> None:
        self.base_path = base_path
        self.name = basename(base_path)
        if run_prefix == "":
            self.run_prefix = self.guess_run_prefix(base_path)
        else:
            self.run_prefix = run_prefix
        self.runs: OrderedDict = self.get_run_paths(base_path, self.run_prefix)

    def __len__(self):
        return len(self.runs)

    @property
    def run_paths(self):
        return [r["run"] for r in self.runs]

    @property
    def out_paths(self):
        return [r["out"] for r in self.runs]

    @property
    def output_folder(self):
        return join(self.base_path, "simulation_runs", "outputs")

    def get_run_items(self, filter: Callable[[Tuple[int, int]], bool] = lambda x: True):
        return [item for item in list(self.runs.items()) if filter(item[0])]

    def get_path(self, run, path_type):
        return self.runs[run][path_type]

    def run_path(self, key):
        return self.get_path(key, "run")

    def out_path(self, key):
        return self.get_path(key, "out")

    def get_sim(self, key: int | Tuple[int, int], id_offset: int = 0) -> Simulation:
        """Get a Simulation object based the suq-controller (par_id, run_id)
        where par_id := one parameter variation run_id := one repetition (different seeds).
        If only and integer is provided as key the run_id defaults to 0.

        Args:
            key int|Tuple[int, int]: request item key. run_id defaults to 0 if not present.

        Returns:
            Simulation: _description_
        """
        if isinstance(key, int):
            key = (key, 0)
        return self.get_run_as_sim(key, id_offset)

    def get_run_as_sim(self, key, id_offset: int = 0):
        run = self.runs[key]
        ctx = RunContext.from_path(join(run["run"], "runContext.json"))
        lbl = f"{self.name}_{self.run_prefix}_{key[0]}_{key[1]}"
        print(lbl)
        return Simulation.from_context(ctx, lbl, id_offset)

    def get_run_context(
        self, key: int | Tuple[int, int], ctx_file_name: str = "runContext.json"
    ):
        if isinstance(key, int):
            key = (key, 0)
        run = self.runs[key]
        return RunContext.from_path(join(run["run"], ctx_file_name))

    def get_simulations(self):
        return [self.get_run_as_sim(k) for k in self.runs.keys()]

    def sim_iter(self, id_offset: int = 0) -> Iterable[Simulation]:
        for k in self.runs.keys():
            yield self.get_run_as_sim(k, id_offset=id_offset)

    def get_simulation_dict(self, lbl_key=False):

        ret = {k: self.get_run_as_sim(k) for k in self.runs.keys()}

        if lbl_key:
            ret = {v.label: v for _, v in ret.items()}

        return OrderedDict(sorted(ret.items(), key=lambda i: (i[0][0], i[0][1])))

    def path(self, *path):
        return os.path.join(self.base_path, *path)

    @classmethod
    def rerun_postprocessing(cls, path: str, jobs=4, log=False, **kwargs):
        run: SuqcStudy = cls(path)
        args = []
        for sim in run.get_simulations():
            _arg = sim.run_context.create_postprocessing_args()
            log_file = os.path.join(sim.run_context.cwd, "log.out")
            if kwargs["failed_only"]:
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as fd:
                        for line in fd.readlines():
                            if "Traceback (most recent call last):" in line:
                                _arg["log"] = log
                                args.append(_arg)
                                break
            else:
                _arg["log"] = log
                args.append(_arg)
        with get_context("spawn").Pool(processes=jobs) as pool:
            ret = pool.map(func=RunContext.exec_runscript, iterable=args)
        return all(ret)

    @classmethod
    def rerun_simulations(
        cls,
        path: str,
        jobs: int = 4,
        what="failed",
        list_only: bool = False,
        filter="all",
        **kwargs,
    ):
        study: SuqcStudy = cls(path)
        # filter runs which should be executed again
        if what == "failed":
            runs: List[RunContext] = study.get_failed_missing_runs()
        else:
            runs: List[RunContext] = [
                study.get_run_context(k) for k in study.runs.keys()
            ]

        if filter != "all":
            # assume run_id==0 for all runs (true for Opp based)
            par_ids = [r.par_id for r in runs]
            filter_ids = apply_str_filter(filter, par_ids)
            runs = [run for run in runs if run.par_id in filter_ids]

        for r in runs:
            print(r.sample_name)
        print(f"found: {len(runs)} failed runs (with filter: {filter})")
        if list_only:
            return True

        args = []
        for r in runs:
            _arg = r.create_run_config_args()
            _arg["log"] = join(r.cwd, "runscript.out")
            _arg["clean_dir"] = r.resultdir
            args.append(dict(args=_arg))

        ts = it.default_timer()
        print(f"spwan {jobs} jobs.")
        ret = run_kwargs_map(
            RunContext.exec_runscript,
            kwargs_iter=args,
            pool_size=jobs,
            raise_on_error=False,
        )
        print(f"Study: took {(it.default_timer() - ts)/60:2.4f} minutes")
        ret = [r for r, _ in ret]
        return all(ret)

    def rename_data_root(self, new_data_root: str, revert: bool = False):
        """Rename data root folder and all existing  runContext.json containing an
        absolute path.

        Args:
            new_data_root (str): new absolute path
        """
        # ensure absolute
        if os.path.abspath(new_data_root) != new_data_root:
            raise ValueError("Expected absolute path")
        if new_data_root.endswith(os.sep):
            new_data_root = new_data_root[:-1]

        _ctx = "runContext.json"
        old_data_root: str = self.base_path
        if old_data_root.endswith(os.sep):
            old_data_root = old_data_root[0:-1]
        context_json = []
        new_runs = OrderedDict()
        for key, value in self.runs.items():
            _run = value["run"].replace(old_data_root, new_data_root)
            _out = value["out"].replace(old_data_root, new_data_root)
            new_runs.setdefault(key, {"run": _run, "out": _out})
            context_json.append(
                (os.path.join(value["run"], _ctx), os.path.join(_run, _ctx))
            )
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H%M")
        if revert:
            for ctx_old, ctx_new in context_json:
                ctx_old_base = os.path.dirname(ctx_old)
                ctx_old_name = os.path.basename(ctx_old)
                backup = glob(f"{ctx_old}_bak*", recursive=False)
                if len(backup) == 0:
                    print("no backup found for {ctx_old}")
                    continue
                elif len(backup) > 1:
                    print("fount multiple backups for {ctx_old}. Skipping...")
                    continue
                print(f"revert to {backup[0]}")
                shutil.move(src=backup[0], dst=ctx_old)
        else:
            for ctx_old, ctx_new in context_json:
                ctx_old_base = os.path.dirname(ctx_old)
                ctx_old_name = os.path.basename(ctx_old)
                ctx_bak = os.path.join(ctx_old_base, f"{ctx_old_name}_bak{now}")
                if os.path.exists(ctx_bak):
                    raise ValueError("Backup file already exists")
                print(f"create backup {ctx_bak}")
                new_lines = []
                with open(ctx_old, "r", encoding="utf-8") as fd:
                    for line in fd.readlines():
                        new_lines.append(line.replace(old_data_root, new_data_root))
                        # print(new_lines[-1])

                shutil.copyfile(src=ctx_old, dst=ctx_bak)
                if not os.path.exists(ctx_bak):
                    raise ValueError(f"Error while creating backup {ctx_bak}")
                with open(ctx_old, "w", encoding="utf-8") as fd:
                    fd.writelines(new_lines)
                new_lines = []

    def update_run_map(
        self,
        run_map: RunMap,
        sim_per_group: int,
        id_offset: int,
        sim_group_factory: SimulationGroupFactory,
        allow_new_groups: bool = True,
        id_filter: Callable[[Tuple[int, int]], bool] = lambda x: True,
        attr: dict | None = None,
    ) -> RunMap:
        """Update given run_map object with simulations contained in this run.

        Args:
            run_map (RunMap): Empty or possible filled run_map object. This object will be mutated.
            sim_per_group (int): number of simulations which belong to the same parameter variation (i.e. only differ in the seed value)
            id_offset (int): In case the RunMap contains simulations form multiple SuqcStudies the id offset ensures unique simulation ids in one RunMap.
            lbl_f (Callable[[Simulation], Any]): Function to create the label. Note the first simulation for each group used to create the label. It is assumed that each
                                                 Simulation object in one group would create the same label.
            allow_new_groups (bool, optional): In case the run_map object was already populated by a previous SuqcStudy this ensures that no new groups are created. Defaults to True
            id_filter (_type_, optional): Function to filter ids of the SuqcStudy *BEFORE* a Simulation object is created. The user must ensure that after the filter the number of
                                          runs is devisable by the sim_per_group method argument. Defaults to lambda x:True (i.e select all).

        Raises:
            ValueError: Number or runs not devisable by sim_per_group
            ValueError: New group_name found in case all_new_groups is False

        Returns:
            RunMap: Updated RunMap object. The run_map method argument is mutated.
        """
        run_items = np.array(self.get_run_items(filter=id_filter))
        if len(run_items) % sim_per_group != 0:
            raise ValueError(
                f"Number of runs is not divisible by sim_per_group {len(run_items)}/{sim_per_group}. check id_filter function or sim_per_group count."
            )
        groups = run_items.reshape((-1, sim_per_group, 2))

        for idx in range(groups.shape[0]):
            group = groups[idx]
            simulations: List[Simulation] = [
                self.get_sim(item[0], id_offset) for item in group
            ]
            sim_group: SimulationGroup = sim_group_factory(
                simulations[0], data=simulations, attr={} if attr is None else attr
            )
            group_name = sim_group.group_name
            if group_name not in run_map and not allow_new_groups:
                raise ValueError(
                    f"RunMap is in append mode only. New group_name '{group_name}' found. Expected: [{run_map.keys()}]"
                )
            run_map.append_or_add(sim_group)

        return run_map
