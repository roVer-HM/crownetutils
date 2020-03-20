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

from roveranalyzer.uitls.file import read_lines

from .configuration import Config


class OppDict(dict):
    """
    run1
      -> runattr
      -> param
      -> mod1
      -> mod2
         -> module
         -> name
         -> title
         -> interpolatin...
      -> mod3
    """

    def __init__(self, df):
        self._df = df

    def __getitem__(self, k):
        return super().__getitem__(k)

    def __setitem__(self, k, v) -> None:
        raise NotImplemented("Read Only Access")

    def __delitem__(self, v) -> None:
        raise NotImplemented("Read Only Access")


def parse_if_number(s):
    try:
        return float(s)
    except:
        return True if s == "true" else False if s == "false" else s if s else None


def parse_ndarray(s):
    return np.fromstring(s, sep=" ") if s else None


def simsec_per_sec(df, ax=None):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot('time', 'simsec_per_sec', data=df, marker='.', linewidth=0)
    ax.set_ylabel('[sim s/s]')
    ax.set_yscale('log')
    ax.set_title('Simsec per second')

    if fig is None:
        return ax
    else:
        return fig, ax


def cumulative_messages(df, ax=None, msg=("msg_present", "msg_in_fes")):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for m in msg:
        ax.plot('time', m, data=df)
    ax.set_xlabel('time [s]')
    ax.set_ylabel('number of messages')
    ax.legend()
    ax.set_title('messages in simulation')

    if fig is None:
        return ax
    else:
        return fig, ax


def parse_cmdEnv_outout(path):
    lines = read_lines(path)

    pattern1 = re.compile("^\*\* Event #(?P<event>\d+)\s+t=(?P<time>\S+)\s+Elapsed: (?P<elapsed>\S+?)s\s+\((?P<elapsed_s>.*?)\).*?completed\s+\((?P<completed>.*?)\% total\)")
    pattern2 = re.compile(
        "^.*?Speed:\s+ev/sec=(?P<events_per_sec>\S+)\s+simsec/sec=(?P<simsec_per_sec>\S+)\s+ev/simsec=(?P<elapsed>\S+)")
    pattern3 = re.compile(
        "^.*?Messages:\s+created:\s+(?P<msg_created>\d+)\s+present:\s+(?P<msg_present>\d+)\s+in\s+FES:\s+(?P<msg_in_fes>\d+)")

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
                raise ValueError('ddd')
        elif l.strip().startswith("Speed:"):
            if m := pattern2.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError('ddd')
        elif l.strip().startswith("Messages:"):
            if m := pattern3.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError('ddd')
        else:
            break

    col = list(pattern1.groupindex.keys())
    col.extend(list(pattern2.groupindex.keys()))
    col.extend(list(pattern3.groupindex.keys()))
    df = pd.DataFrame(data, columns=col)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df


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

    @staticmethod
    def _converters():
        return {
            # 'run': ,
            # 'type': ,
            # 'module': ,
            # 'name': ,
            # 'attrname': ,
            "attrvalue": parse_if_number,
            # 'value': ,                    # scalar data
            # 'count': ,                    # scalar data
            # 'sumweights': ,               # scalar data
            # 'mean': ,                     # scalar data
            # 'stddev': ,                   # scalar data
            # 'min': ,                      # scalar data
            # 'max': ,                      # scalar data
            "binedges": parse_ndarray,  # histogram data
            "binvalues": parse_ndarray,  # histogram data
            "vectime": parse_ndarray,  # vector data
            "vecvalue": parse_ndarray,
        }  # vector data

    @classmethod
    def _is_valid(cls, file: str):
        if file.endswith(".sca") or file.endswith(".vec"):
            if os.path.exists(file):
                return True
        return False

    def load_csv(self, csv_file) -> pd.DataFrame:
        """
        #load_csv to load an existing OMNeT++ csv file. The following columns are expected to exist.
          'run', 'type', 'module', 'name', 'attrname', 'attrvalue', 'value', 'count', 'sumweights',
          'mean', 'stddev', 'min', 'max', 'binedges', 'binvalues', 'vectime', 'vecvalue'.
        :param csv_file:    Path to csv file
        :return:            pd.DataFrame with extra namespace 'opp' (an OppAccessor object with helpers)
        """
        df = pd.read_csv(csv_file, converters=self._converters())
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
        self, input_paths: List[str], scave_filter: str = None, recursive=True,
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

        df = pd.read_csv(
            io.BytesIO(stdout), encoding="utf-8", converters=self._converters()
        )
        df = df.opp.pre_process()
        # df.opp.attr["cmd"] = cmd
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
        cmd = self._SCAVE_TOOL
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
            scave_cmd.wait(self.timeout)
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
