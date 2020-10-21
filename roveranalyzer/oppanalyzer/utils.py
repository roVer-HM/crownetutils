import contextlib
import glob
import io
import logging
import os
import pprint as pp
import re
import signal
import subprocess
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from roveranalyzer.oppanalyzer.configuration import Config
from roveranalyzer.uitls import PathHelper, Timer
from roveranalyzer.uitls.file import read_lines
from roveranalyzer.vadereanalyzer.scenario_output import ScenarioOutput


def stack_vectors(
    df,
    index,
    columns=("vectime", "vecvalue"),
    col_data_name="data",
    drop=None,
    time_as_index=False,
):
    """
    data frame which only contains opp vector rows.
    """
    timer = Timer.create_and_start("set index", label=stack_vectors.__name__)
    __df = df.set_index(index)

    timer.stop_start("stack data")
    stacked = list()
    for col in columns:
        stacked.append(__df[col].apply(pd.Series).stack())

    timer.stop_start("concatenate data")
    __df = pd.concat(stacked, axis=1, keys=columns)

    timer.stop_start("cleanup index")  #
    if None in __df.index.names:
        __df.index = __df.index.droplevel(level=None)

    timer.stop_start("drop columns or index level")
    for c in drop:
        if c in __df.index.names:
            __df.index = __df.index.droplevel(level=c)
        elif c in __df.columns:
            __df = __df.drop(c, axis=1)
        # else:
        #     print(f"waring: given name {c} cannot be droped. Does not exist.")

    timer.stop_start("rename vecvalue")
    __df = __df.rename({"vectime": "time", "vecvalue": col_data_name}, axis=1)

    if time_as_index:
        __df = __df.set_index("time", append=True)

    __df = __df.sort_index()
    timer.stop()
    return __df


def build_time_series(
    opp_df,
    opp_vector_names,
    opp_vector_col_names=None,
    opp_index=("run", "module", "name"),
    opp_drop=("name", "module"),
    hdf_store=None,
    hdf_key=None,
    time_bin_size=0.0,
    index=None,
    fill_na=None,
):
    """
    Build normalized data frames for OMNeT++ vectors.
    opp_df:         data frame from OMNeT++
    opp_vectors:    list of vector names to concatenate. The vectime axis of these vectors are merged.
    opp_vector_names:      same length as opp_vectors containing the column names for used for vecvalues
    opp_index:      columns of opp_df used for uniqueness
    opp_drop:       index to drop after stacking of the data frame
    hdf_store:      HDF store used to save generated data frame. Default=None (do not save to disk)
    hdf_key:        key to use for HDF storage.
    time_bin_size:       size of time bins used for time_step index. Default=0.0 (do not create a time_step index)
    index:          Reindex result with given index. Default=None (just leave the given index)
    fill_na:        fill created N/A values with given.
    """
    # "dcf.channelAccess.inProgressFrames.["queueingTime:vector", "queueingLength:vector"]"
    # "dcf.channelAccess.pendingQueue.["queueingTime:vector", "queueingLength:vector"]"

    # check input
    timer = Timer.create_and_start("check input", label=build_time_series.__name__)
    if opp_vector_col_names is None:
        opp_vector_col_names = [
            f"val_{idx}" for idx in np.arange(0, len(opp_vector_names))
        ]
    if len(opp_vector_names) != len(opp_vector_col_names):
        raise ValueError(f"opp_vectors length does not match with opp_vectors")
    if hdf_store is not None and hdf_key is None:
        raise ValueError(f"a hdf store is given but hdf_key is missing")
    if index is not None and "time_step" in index and time_bin_size <= 0.0:
        raise ValueError(
            f"return index contains 'time_step' but time_bin_size is not set."
        )

    data = []

    for idx, c in enumerate(opp_vector_names):
        timer.stop_start(f"stack {c} as column {opp_vector_col_names[idx]}")
        _df = opp_df.opp.filter().vector().name(c).apply()
        _df = stack_vectors(
            _df,
            index=list(opp_index),
            drop=list(opp_drop),
            col_data_name=opp_vector_col_names[idx],
            time_as_index=True,
        )
        data.append(_df)

    print(f"concatenating {len(data)} data frames. This will take some time...")
    timer.stop_start(f"concatenating {len(data)} data frames. done")
    _df_ret = pd.concat(data, axis=1)
    if fill_na is not None:
        _df_ret = _df_ret.fillna(fill_na)

    if time_bin_size > 0.0:
        timer.stop_start(f"add time_step index with bin size {time_bin_size}")
        time_idx = _df_ret.index.get_level_values("time")
        # create bins based on given time_bin_size
        bins = np.arange(
            np.floor(time_idx.min()), np.ceil(time_idx.max()), step=time_bin_size
        )
        time_bin = pd.Series(
            np.digitize(time_idx, bins) * time_bin_size + time_idx.min(),
            name="time_step",
        )
        time_bin.index = _df_ret.index
        _df_ret = pd.concat([_df_ret, time_bin], axis=1)

        _df_ret = _df_ret.set_index("time_step", append=True)

    if index is not None:
        timer.stop_start(f"apply index {index} and sort")
        _df_ret = _df_ret.reset_index()
        _df_ret = _df_ret.set_index(index)
        _df_ret = _df_ret.sort_index()

    timer.stop()
    return _df_ret


def simsec_per_sec(df, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot("time", "simsec_per_sec", data=df, marker=".", linewidth=0)
    ax.set_ylabel("[sim s/s]")
    ax.set_yscale("log")
    ax.set_title("Simsec per second")

    if fig is None:
        return ax
    else:
        return fig, ax


def cumulative_messages(
    df,
    ax=None,
    msg=("msg_present", "msg_in_fes"),
    lbl=("number of messages", "messages in fes"),
    set_lbl=True,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for idx, m in enumerate(msg):
        ax.plot("time", m, data=df, label=lbl[idx])

    if set_lbl:
        ax.set_xlabel("time [s]")
        ax.set_ylabel("number of messages")
        ax.legend()
        ax.set_title("messages in simulation")

    if fig is None:
        return ax
    else:
        return fig, ax


def parse_cmdEnv_outout(path):
    lines = read_lines(path)

    pattern1 = re.compile(
        "^\*\* Event #(?P<event>\d+)\s+t=(?P<time>\S+)\s+Elapsed: (?P<elapsed>\S+?)s\s+\((?P<elapsed_s>.*?)\).*?completed\s+\((?P<completed>.*?)\% total\)"
    )
    pattern2 = re.compile(
        "^.*?Speed:\s+ev/sec=(?P<events_per_sec>\S+)\s+simsec/sec=(?P<simsec_per_sec>\S+)\s+ev/simsec=(?P<elapsed>\S+)"
    )
    pattern3 = re.compile(
        "^.*?Messages:\s+created:\s+(?P<msg_created>\d+)\s+present:\s+(?P<msg_present>\d+)\s+in\s+FES:\s+(?P<msg_in_fes>\d+)"
    )

    data = []
    event_data = []
    for l in lines:
        if l.strip().startswith("** "):
            if len(event_data) != 0:
                data.append(event_data)
            event_data = []
            if m := pattern1.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        elif l.strip().startswith("Speed:"):
            if m := pattern2.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        elif l.strip().startswith("Messages:"):
            if m := pattern3.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        else:
            break

    col = list(pattern1.groupindex.keys())
    col.extend(list(pattern2.groupindex.keys()))
    col.extend(list(pattern3.groupindex.keys()))
    df = pd.DataFrame(data, columns=col)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


class ScaveConverter:
    """
    pandas csv to DataFrame converter. Provides a dict of functions to use while
    reading csv file. The keys in the dict must match the column names.
    """
    def __init__(self):
        pass

    def parse_if_number(self, s):
        try:
            return float(s)
        except:
            return True if s == "true" else False if s == "false" else s if s else None

    def parse_ndarray(self, s):
        return np.fromstring(s, sep=" ", dtype=float) if s else None

    def parse_series(self, s):
        return pd.Series(np.fromstring(s, sep=" ", dtype=float)) if s else None

    def get_series_parser(self):
        return {
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_series,  # histogram data
            "binvalues": self.parse_series,  # histogram data
            "vectime": self.parse_series,  # vector data
            "vecvalue": self.parse_series,
        }

    def get_array_parser(self):
        return {
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_ndarray,  # histogram data
            "binvalues": self.parse_ndarray,  # histogram data
            "vectime": self.parse_ndarray,  # vector data
            "vecvalue": self.parse_ndarray,
        }

    def get(self):
        return self.get_array_parser()


class ScaveRunConverter(ScaveConverter):
    """
    pandas csv to DataFrame converter. Provides a dict of functions to use while
    reading csv file. The keys in the dict must match the column names.

    Simplify run name by providing a shorter name
    """
    def __init__(self, run_short_hand="r"):
        super().__init__()
        self._short_hand = run_short_hand
        self.run_map = {}
        self.network_map = {}

    def parse_run(self, s):
        if s in self.run_map:
            return self.run_map[s]
        else:
            ret = f"{self._short_hand}_{len(self.run_map)}"
            self.run_map.setdefault(s, ret)
            return ret

    def mapping_data_frame(self):
        d_a = [["run", k, v] for k, v in self.run_map.items()]
        return pd.DataFrame(d_a, columns=["level", "id", "mapping"])

    def get(self):
        return self.get_array_parser()

    def get_array_parser(self):
        return {
            "run": self.parse_run,
            "attrvalue": self.parse_if_number,
            "binedges": self.parse_ndarray,  # histogram data
            "binvalues": self.parse_ndarray,  # histogram data
            "vectime": self.parse_ndarray,  # vector data
            "vecvalue": self.parse_ndarray,
        }


class Suffix:
    HDF = ".h5"
    CSV = ".csv"
    PNG = ".png"
    PDF = ".pdf"
    DIR = ".d"


class HdfProvider:
    """
    Wrap access to a given HDF store (hdf_path) in a context manager. Wrapper is lazy and checks if store exist
    are *Not* done. Caller must ensure file exists
    """

    def __init__(self, hdf_path):
        self._hdf_path = hdf_path
        self._hdf_args = {"complevel": 9, "complib": "zlib"}

    @contextlib.contextmanager
    def ctx(self, mode="a", **kwargs) -> pd.HDFStore:
        _args = dict(self._hdf_args)
        _args.update(kwargs)
        store = pd.HDFStore(self._hdf_path, mode=mode, **_args)
        try:
            yield store
        finally:
            store.close()

    def set_args(self, append=False, **kwargs):
        if append:
            self._hdf_args.update(kwargs)
        else:
            self._hdf_args = kwargs

    def get_data(self, key):
        with self.ctx(mode="r") as store:
            df = store.get(key=key)
        return df

    def exists(self):
        """ check for HDF store """
        return os.path.exists(self._hdf_path)

    def has_key(self, key):
        """
        check if key exists in HDF Store. True if yes, False if key OR store does not exist
        """
        if self.exists():
            with self.ctx(mode="r") as store:
                return key in store
        return False


class OppDataProvider:
    """
    provide dataframe from OMneT++ output. The source of the data my be:
    - *.sca or *.vec files,
    - already processed *.csv files provided by some earlier run of the a ScaveTool or
    - a HDF-Store containing preprocessed dataframes.
    """
    def __init__(self, path: PathHelper, analysis_name, analysis_dir=None, hdf_store_name=None, cfg: Config = None):
        self._root = path
        # output
        self._analysis_name = analysis_name
        self._analysis_dir = analysis_dir
        if analysis_dir is None:
            self._analysis_dir = f"{analysis_name}{Suffix.DIR}"

        # ScaveTool
        self._scave_filter = ""
        self._opp_input_paths = []
        self._converter = ScaveRunConverter(run_short_hand="r")
        self._cfg = cfg

        # initialize HDF store
        _hdf_store_name = hdf_store_name
        if _hdf_store_name is None:
            _hdf_store_name = f"{analysis_name}{Suffix.HDF}"
        self._hdf_store = HdfProvider(self._root.join(self._analysis_dir, _hdf_store_name))

        self._root.make_dir(self._analysis_dir, exist_ok=True)

    @property
    def root(self):
        return self._root

    @property
    def out_dir(self):
        return self._root.join(self._analysis_dir)

    @property
    def csv_path(self):
        return self._root.join(self._analysis_dir, f"{self._analysis_name}{Suffix.CSV}")

    @property
    def hdf_store(self):
        return self._hdf_store

    def set_scave_filter(self, *scave_filters, append=False, operator="AND"):
        _op = f" {operator} "
        _filter = list(scave_filters)
        if append:
            _filter.insert(0, f"({self._scave_filter})")
            self._scave_filter = _op.join(_filter)
        else:
            self._scave_filter = _op.join(list(scave_filters))
        return self

    def get_scave_filter(self):
        return self._scave_filter

    def set_scave_input_path(self, *input_path, append=False, rel_path=True):
        if rel_path:
            _paths = [self._root.join(p) for p in input_path]
        else:
            _paths = list(input_path)
        if append:
            self._opp_input_paths.extend(_paths)
        else:
            self._opp_input_paths = _paths
        return self

    def get_scave_input_path(self):
        return self._opp_input_paths

    def set_converter(self, converter):
        self._converter = converter

    def get_converter(self):
        return self._converter

    def df_from_csv(self, override=False, recursive=True):
        if self._cfg is None:
            _scv = ScaveTool()
        else:
            _scv = ScaveTool(config=self._cfg)
        _csv = _scv.create_or_get_csv_file(
            csv_path=self.csv_path,
            input_paths=self.get_scave_input_path(),
            scave_filter=self.get_scave_filter(),
            override=override,
            recursive=recursive,
        )
        return _scv.load_csv(_csv, converters=self._converter)

    def save_converter_to_hdf(self, key, mode="a", **kwargs):
        with self.hdf_store.ctx(mode=mode, **kwargs) as store:
            self._converter.mapping_data_frame().to_hdf(store, key=key)

    def save_to_output(self, fig: plt.Figure, file_name, **kwargs):
        fig.savefig(os.path.join(self.out_dir, file_name), **kwargs)

    def vadere_output_from(self, run_dir, is_abs=False):
        if is_abs:
            return ScenarioOutput.create_output_from_project_output(run_dir)
        g = glob.glob(self._root.join(run_dir, "**/*.scenario"))
        if len(g) != 1:
            raise ValueError(f"expected a single scenario file got: {'/n'.join(g)}")
        vout_dir = os.path.dirname(g[0])
        return ScenarioOutput.create_output_from_project_output(vout_dir)


class ScaveTool:
    """
    Python wrapper for OMNeT++ scavetool.

    Allows simple access to query and export functions defined in the scavetool.
    See #print_help, #print_export_help and #print_filter_help for scavetool usage.

    Use #create_or_get_csv_file to create (or use existing) csv files from one or
    many OMNeT++ result files. The method  accepts multiple glob patters which are
    search recursive (default) for files ending in *.vec and *.sca.
    If given a scave_filter is applied to reduce the amount of imported data. See #print_print_filter_help
    on usage.

    Use #load_csv to load an existing OMNeT++ csv file. The following columns are expected to exist.
      'run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value', 'count', 'sumweights',
      'mean', 'stddev', 'min', 'max', 'binedges', 'binvalues', 'vectime', 'vecvalue'.
    """

    _EXPORT = "x"
    _QUERY = "q"
    _INDEX = "i"
    _OUTPUT = "-o"
    _FILTER = "--filter"

    def __init__(self, config: Config = None, timeout=360):
        if config is None:
            self._config = Config()  # default
        else:
            self._config = config
        self._SCAVE_TOOL = self._config.scave_cmd
        self.timeout = timeout

    @classmethod
    def _is_valid(cls, file: str):
        if file.endswith(".sca") or file.endswith(".vec"):
            if os.path.exists(file):
                return True
        return False

    def load_csv(self, csv_file, converters=None) -> pd.DataFrame:
        """
        #load_csv to load an existing OMNeT++ csv file. The following columns are expected to exist.
          'run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value', 'count', 'sumweights',
          'mean', 'stddev', 'min', 'max', 'binedges', 'binvalues', 'vectime', 'vecvalue'.
        :param csv_file:    Path to csv file
        :return:            pd.DataFrame with extra namespace 'opp' (an OppAccessor object with helpers)
        """
        if converters is None:
            converters = ScaveConverter()
        df = pd.read_csv(csv_file, converters=converters.get())
        # df.opp.attr["csv_path"] = csv_file
        return df

    def create_or_get_csv_file(
        self,
        csv_path,
        input_paths: List[str],
        override=False,
        scave_filter: str = None,
        recursive=True,
        print_selected_files=True,
    ):
        """
        #create_or_get_csv_file to create (or use existing) csv files from one or
        many OMNeT++ result files. The method  accepts multiple glob patters which are
         search recursive (default) for files ending in *.vec and *.sca.
        If given a scave_filter is applied to reduce the amount of imported data. See #print_print_filter_help
        on usage.
        :param csv_path:             path to existing csv file or path to new csv file (see :param override)
        :param input_paths:          List of glob patters search for *.vec and *.sca files
        :param override:             (default: False) override existing csv_path
        :param scave_filter:         (default: None) string based filter for scavetool see #print_filter_help for syntax
        :param recursive:            (default: True) use recursive glob patterns
        :param print_selected_files: print list of files selected by the given input_paths.
        :return:
        """
        if os.path.isfile(csv_path) and not override:
            return os.path.abspath(csv_path)

        cmd = self.export_cmd(
            input_paths=input_paths,
            output=os.path.abspath(csv_path),
            scave_filter=scave_filter,
            recursive=recursive,
            print_selected_files=print_selected_files,
        )
        self.exec(cmd)

        return os.path.abspath(csv_path)

    def load_df_from_scave(
        self,
        input_paths: List[str],
        scave_filter: str = None,
        recursive=True,
        converters=None,
    ) -> pd.DataFrame:
        """
         Directly load data into Dataframe from *.vec and *.sca files without creating a
         csv file first. Use stdout of scavetool to create Dataframe.

         Helpful variant for automated scripts to reduce memory footprint.

        :param input_paths:     List of glob patters search for *.vec and *.sca files
        :param scave_filter:    (default: None) string based filter for scavetool see #print_filter_help for syntax
        :param recursive:       (default: True) use recursive glob patterns
        :return:
        """
        cmd = self.export_cmd(
            input_paths=input_paths,
            output="-",  # read from stdout of scavetool
            scave_filter=scave_filter,
            recursive=recursive,
            options=["-F", "CSV-R"],
        )
        stdout, stderr = self.read_stdout(cmd)
        if stdout == b"":
            logging.error("error executing scavetool")
            print(str(stderr, encoding="utf8"))
            return pd.DataFrame()

        if converters is None:
            converters = ScaveConverter()
        # skip first row (container output)
        df = pd.read_csv(
            io.BytesIO(stdout), encoding="utf-8", skiprows=1, converters=converters.get()
        )
        return df

    def export_cmd(
        self,
        input_paths,
        output,
        scave_filter=None,
        recursive=True,
        options=None,
        print_selected_files=False,
    ):
        cmd = self._SCAVE_TOOL[:]
        cmd.append(self._EXPORT)
        cmd.append(self._OUTPUT)
        cmd.append(output)
        if scave_filter is not None:
            cmd.append(self._FILTER)
            cmd.append(self._config.escape(scave_filter))

        if options is not None:
            cmd.extend(options)

        if len(input_paths) == 0:
            raise ValueError("no *.vec or *.sca files given.")

        opp_result_files = list()
        for file in input_paths:
            opp_result_files.extend(glob.glob(file, recursive=recursive))

        opp_result_files = [
            f for f in opp_result_files if f.endswith(".vec") or f.endswith(".sca")
        ]
        if len(opp_result_files) == 0:
            raise ValueError("no opp input files selected.")

        log = "\n".join(opp_result_files)
        logging.info(f"found *.vec and *.sca:\n {log}")
        if print_selected_files:
            print("selected files:")
            for f in opp_result_files:
                print(f"\t{f}")

        cmd.extend(opp_result_files)
        return cmd

    def print_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append("--help")
        self.exec(cmd)

    def print_export_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append(self._EXPORT)
        cmd.append("--help")
        self.exec(cmd)

    def print_filter_help(self):
        cmd = self._SCAVE_TOOL
        cmd.append("help")
        cmd.append("filter")
        self.exec(cmd)

    def read_stdout(self, cmd):
        scave_cmd = subprocess.Popen(
            cmd,
            cwd=os.path.curdir,
            stdin=None,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            out, err = scave_cmd.communicate(timeout=self.timeout)
            return out, err
        except subprocess.TimeoutExpired:
            logging.error("Timout reached")
            scave_cmd.kill()
            return b"", io.StringIO("timeout reached")

    def exec(self, cmd):
        scave_cmd = subprocess.Popen(
            cmd,
            cwd=os.path.curdir,
            shell=False,
            stdin=None,
            env=os.environ.copy(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            scave_cmd.wait()
            if scave_cmd.returncode != 0:
                logging.error(f"return code was {scave_cmd.returncode}")
                logging.error("command:")
                logging.error(f"{pp.pprint(cmd)}")
                print(scave_cmd.stdout.read().decode("utf-8"))
                print(scave_cmd.stderr.read().decode("utf-8"))

            else:
                logging.info(f"return code was {scave_cmd.returncode}")
                print(scave_cmd.stdout.read().decode("utf-8"))

        except subprocess.TimeoutExpired:
            logging.info(f"scavetool timeout reached. Kill it")
            os.kill(scave_cmd.pid, signal.SIGKILL)
            time.sleep(0.5)
            if scave_cmd.returncode is None:
                logging.error("scavetool still not dead after SIGKILL")
                raise

        logging.info(f"return code: {scave_cmd.returncode}")


class StatsTool:
    """
    Toolset for calculating and nicely printing statistics
    """

    @staticmethod
    def stats_table(data, unit: str = "", name: str = "") -> str:
        """
        Create a table listing the most important statistic values

        :param data:    data to calculate the statistics on
        :param unit:    SI unit of data (optional)
        :param name:    name of the data to be printed (optional)
        :return:        string with statistics table
        """
        table = "=============================================================\n"
        if len(name) > 0:
            table += (f"! Data: {name:51} !\n"
                      "-------------------------------------------------------------\n"
                      )

        table += (
            f"! nr of values : {len(data):15}                            !\n"
            f"! arith. mean  : {np.mean(data):15.6f} {unit:>4}                       !\n"
            f"! minimum      : {np.min(data):15.6f} {unit:>4}                       !\n"
            f"! maximum      : {np.max(data):15.6f} {unit:>4}                       !\n"
            f"! median       : {np.median(data):15.6f} {unit:>4}                       !\n"
            f"! std. dev.    : {np.std(data):15.6f} {unit:>4}                       !\n"
            f"! variance     : {np.var(data):15.6f} {unit:>4}^2                     !\n"
            "=============================================================\n"
        )

        return table


class PlotAttrs:
    """
    PlotAttrs is a singleton guaranteeing unique plot parameters
    """
    
    class __PlotAttrs:
        plot_marker = [".", "*", "o", "v", "1", "2", "3", "4"]
        plot_color = ["b", "g", "r", "c", "m", "y", "k", "w"]

        def __init__(self):
            pass

    instance: object = None

    def __init__(self):
        if not PlotAttrs.instance:
            PlotAttrs.instance = PlotAttrs.__PlotAttrs()
        self.idx_m = -1
        self.idx_c = -1

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def get_marker(self) -> str:
        ret = self.instance.plot_marker[self.idx_m]
        self.idx_m += 1
        if self.idx_m >= len(self.instance.plot_marker):
            self.idx_m = 0
        return ret

    def get_color(self) -> str:
        ret = self.instance.plot_color[self.idx_c]
        self.idx_c += 1
        if self.idx_c >= len(self.instance.plot_color):
            self.idx_c = 0
        return ret

    def reset(self):
        self.idx_c = 0
        self.idx_m = 0


class Simulation(object):
    """
    The Simulation class specifies the information required to read and plot a simulation scenario.

    :param id:          Unique ID of the scenario
    :param path:        Path to all *.vec and *.sca files produced by the scenario
    :param description: Human readable description used in plot legend for this scenario
    """
    def __init__(self, id: str, path: str, description: str):
        self.id = id
        self.path = path
        self.desc = description
