import glob
import io
import os
import pprint as pp
import signal
import subprocess
import time
from typing import List, Union

import numpy as np
import pandas as pd

from roveranalyzer.simulators.opp.configuration import Config
from roveranalyzer.utils import Timer, logger


class ScaveData:
    @classmethod
    def build_time_series(
        cls,
        opp_df,
        opp_vector_names,
        hdf_store=None,
        hdf_key=None,
        time_bin_size=0.0,
    ):
        """
        Build normalized data frames for OMNeT++ vectors. See Opp.normalize_vectors
        opp_df:         data frame from OMNeT++
        opp_vector_names:      same length as opp_vectors containing the column names for used for vecvalues
        hdf_store:      HDF store used to save generated data frame. Default=None (do not save to disk)
        hdf_key:        key to use for HDF storage.
        time_bin_size:       size of time bins used for time_step index. Default=0.0 (do not create a time_step index)
        """
        # "dcf.channelAccess.inProgressFrames.["queueingTime:vector", "queueingLength:vector"]"
        # "dcf.channelAccess.pendingQueue.["queueingTime:vector", "queueingLength:vector"]"

        # check input
        timer = Timer.create_and_start(
            "check input", label=cls.build_time_series.__name__
        )

        _df_ret = (
            opp_df.opp.filter()
            .vector()
            .name_in(opp_vector_names)
            .normalize_vectors(axis=0)
        )

        if time_bin_size > 0.0:
            timer.stop_start(f"add time_step index with bin size {time_bin_size}")
            time_idx = _df_ret["time"]
            # create bins based on given time_bin_size
            bins = np.arange(
                np.floor(time_idx.min()), np.ceil(time_idx.max()), step=time_bin_size
            )
            time_bin = pd.Series(
                np.digitize(time_idx, bins) * time_bin_size + time_idx.min(),
                name="time_step",
            )
            _df_ret["time_step"] = time_bin
            _df_ret = _df_ret.drop(columns=["time"])
            _df_ret = _df_ret.set_index("time_step", drop=True)

        timer.stop()
        return _df_ret

    @classmethod
    def stack_vectors(
        cls,
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
        timer = Timer.create_and_start("set index", label=cls.stack_vectors.__name__)
        df = df.set_index(index)

        timer.stop_start("stack data")
        stacked = list()
        for col in columns:
            stacked.append(df[col].apply(pd.Series).stack())

        timer.stop_start("concatenate data")
        df = pd.concat(stacked, axis=1, keys=columns)

        timer.stop_start("cleanup index")  #
        if None in df.index.names:
            df.index = df.index.droplevel(level=None)

        timer.stop_start("drop columns or index level")
        for c in drop:
            if c in df.index.names:
                df.index = df.index.droplevel(level=c)
            elif c in df.columns:
                df = df.drop(c, axis=1)
            # else:
            #     print(f"waring: given name {c} cannot be droped. Does not exist.")

        timer.stop_start("rename vecvalue")
        df = df.rename({"vectime": "time", "vecvalue": col_data_name}, axis=1)

        if time_as_index:
            df = df.set_index("time", append=True)

        df = df.sort_index()
        timer.stop()
        return df


class ScaveFilter:
    """
    Build opp_scavetool filter using builder pattern.
    """

    @classmethod
    def create(cls):
        return cls()

    def __init__(self, config: Config = None):
        if config is None:
            self._config = Config()  # default
        else:
            self._config = config
        self._filter = []
        self._groups = 0

    def gOpen(self):
        self._filter.append("(")
        self._groups += 1
        return self

    def gClose(self):
        self._filter.append(")")
        self._groups -= 1
        if self._groups < 0:
            raise ValueError(
                f"Scave filter group mismatch. Closed one group that was not "
                f"opened: '{' '.join(self._filter)}'"
            )
        return self

    @staticmethod
    def _breaket_workaround(val):
        val_old = val
        val = val = val.replace(r"(", r"?")
        val = val = val.replace(r")", r"?")
        if val_old != val:
            print(
                "warning: using breaket workaround due to parse issue in omnetpp see #2"
            )
        return val

    def AND(self):
        self._filter.append("AND")
        return self

    def OR(self):
        self._filter.append("OR")
        return self

    def file(self, val):
        self._filter.extend(["file", "=~", val])
        return self

    def run(self, val):
        self._filter.extend(["run", "=~", val])
        return self

    def t_scalar(self):
        self._filter.extend(["type", "=~", "scalar"])
        return self

    def t_vector(self):
        self._filter.extend(["type", "=~", "vector"])
        return self

    def t_statistics(self):
        self._filter.extend(["type", "=~", "statistics"])
        return self

    def t_histogram(self):
        self._filter.extend(["type", "=~", "histogram"])
        return self

    def t_parameter(self):
        self._filter.extend(["type", "=~", "parameter"])
        return self

    def type(self, val):
        self._filter.extend(["type", "=~", val])
        return self

    def name(self, val):
        self._filter.extend(["name", "=~", self._breaket_workaround(val)])
        return self

    def module(self, val):
        self._filter.extend(["module", "=~", val])
        return self

    def runattr(self, name):
        self._filter.append(f"runattr:{name}")
        return self

    def itervar(self, name):
        self._filter.append(f"itervar:{name}")
        return self

    def config(self, name):
        self._filter.append(f"config:{name}")
        return self

    def attr(self, name):
        self._filter.append(f"attr:{name}")
        return self

    def build(self, escape=True):
        if self._groups != 0:
            raise ValueError(
                f"Scave filter group mismatch." f"opened: '{' '.join(self._filter)}'"
            )
        if escape:
            return self._config.escape(" ".join(self._filter))
        else:
            return " ".join(self._filter)

    def str(self):
        return " ".join(self._filter)


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
        self._SCAVE_TOOL = self._config.scave_cmd(silent=True)
        self.timeout = timeout

    @classmethod
    def _is_valid(cls, file: str):
        if file.endswith(".sca") or file.endswith(".vec"):
            if os.path.exists(file):
                return True
        return False

    def filter_builder(self) -> ScaveFilter:
        return ScaveFilter(self._config)

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
        return df

    def create_or_get_csv_file(
        self,
        csv_path,
        input_paths: List[str],
        override=False,
        scave_filter: Union[str, ScaveFilter] = None,
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
        input_paths: Union[str, List[str]],
        scave_filter: Union[str, ScaveFilter] = None,
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
        if type(input_paths) == str:
            input_paths = [input_paths]

        cmd = self.export_cmd(
            input_paths=input_paths,
            output="-",  # read from stdout of scavetool
            scave_filter=scave_filter,
            recursive=recursive,
            options=["-F", "CSV-R"],
        )
        print(" ".join(cmd))
        stdout, stderr = self.read_stdout(cmd, encoding="")
        if stdout == b"":
            logger.error("error executing scavetool")
            print(str(stderr, encoding="utf8"))
            return pd.DataFrame()

        if converters is None:
            converters = ScaveConverter()
        # skip first row (container output)
        df = pd.read_csv(
            io.BytesIO(stdout),
            encoding="utf-8",
            converters=converters.get(),
        )
        return df

    def export_cmd(
        self,
        input_paths,
        output,
        scave_filter: Union[str, ScaveFilter] = None,
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
            if type(scave_filter) == str:
                cmd.append(self._config.escape(scave_filter))
            else:
                cmd.append(scave_filter.build(escape=True))

        if options is not None:
            cmd.extend(options)

        if len(input_paths) == 0:
            raise ValueError("no *.vec or *.sca files given.")

        # todo check if glob pattern exists first only then do this and the check
        opp_result_files = list()
        if any([_f for _f in input_paths if "*" in _f]):
            for file in input_paths:
                opp_result_files.extend(glob.glob(file, recursive=recursive))
        else:
            opp_result_files.extend(input_paths)

        opp_result_files = [
            f for f in opp_result_files if f.endswith(".vec") or f.endswith(".sca")
        ]
        if len(opp_result_files) == 0:
            raise ValueError("no opp input files selected.")

        log = "\n".join(opp_result_files)
        logger.info(f"found *.vec and *.sca:\n {log}")
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

    def read_parameters(self, result_file, scave_filter=None):
        if scave_filter is None:
            scave_filter = self.filter_builder().t_parameter().build()
        cmd = self._config.scave_cmd(silent=True)
        cmd.extend(
            [
                "query",
                "--list-results",
                "--bare",
                "--grep-friendly",
                "--tabs",
                "--filter",
                scave_filter,
                result_file,
            ]
        )
        print(" ".join(cmd))
        out, err = self.read_stdout(cmd)
        if err != "":
            raise RuntimeError(f"container return error: \n{err}")

        out = [line.split("\t") for line in out.split("\n") if line != ""]
        return pd.DataFrame(out, columns=["run", "type", "module", "name", "value"])

    def read_stdout(self, cmd, encoding="utf-8"):
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
            if encoding != "":
                return out.decode(encoding), err.decode(encoding)
            else:
                return out, err
        except subprocess.TimeoutExpired:
            logger.error("Timout reached")
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
                logger.error(f"return code was {scave_cmd.returncode}")
                logger.error("command:")
                logger.error(f"{pp.pprint(cmd)}")
                print(scave_cmd.stdout.read().decode("utf-8"))
                print(scave_cmd.stderr.read().decode("utf-8"))

            else:
                logger.info(f"return code was {scave_cmd.returncode}")
                print(scave_cmd.stdout.read().decode("utf-8"))

        except subprocess.TimeoutExpired:
            logger.info(f"scavetool timeout reached. Kill it")
            os.kill(scave_cmd.pid, signal.SIGKILL)
            time.sleep(0.5)
            if scave_cmd.returncode is None:
                logger.error("scavetool still not dead after SIGKILL")
                raise

        logger.info(f"return code: {scave_cmd.returncode}")


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
