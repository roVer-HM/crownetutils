from __future__ import annotations

from functools import partial
from glob import escape
from typing import Any, Callable, List, Protocol

import pandas as pd
from pandas.io.formats.style import Styler


class EmptyFrameConsumer:
    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        return df


class FrameConsumerList:
    """Class implementing the FrameConsumer protocol where
    multiple FrameConsumers are chained"""

    @classmethod
    def get(cls, *fc):
        return cls(fc)

    def __init__(self, fc_list: List[FrameConsumer]) -> None:
        self.fc_list = fc_list

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        for fc in self.fc_list:
            df = fc(df, *args, **kwargs)
        return df


class FrameConsumer(Protocol):
    """Function that will alter the provided DataFrame in some way. FrameConsumer.EMPTY will do nothing."""

    # the 'do nothing consumer'
    EMPTY: FrameConsumer = EmptyFrameConsumer()

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        pass


def siunitx_format(val, cmd, options=None):
    if options is None:
        return f"\{cmd}{{{val}}}"
    else:
        return f"\{cmd}[{options}]{{{val}}}"


def siunitx(cmd="num", precision: int = 4, *args, **kwargs) -> Callable[[Any], str]:
    options = [f"round-precision={precision}"]
    options.extend(args)
    options.extend([f"{k}={v}" for k, v in kwargs.items()])
    options = ",".join(options)
    return partial(siunitx_format, cmd=cmd, options=options)


def format_frame(df: pd.DataFrame, si_func=lambda x: f"\\num{{{x}}}") -> pd.DataFrame:

    _df: pd.DataFrame = df.copy(deep=True)
    if isinstance(si_func, dict):
        for col, _func in si_func.items():
            if col in _df.columns:
                _df[col] = _df[col].apply(_func)
    else:
        _df = _df.applymap(si_func)

    return _df


def save_as_tex_table(
    df: pd.DataFrame,
    path: str,
    selected_only: bool = True,
    rename: dict | None = None,
    col_format: dict | None = None,
    str_replace: Callable[[str], str] = lambda x: x,
):

    _df: pd.DataFrame = df.copy(deep=True)
    if rename is not None:
        _df = _df.rename(columns=rename)
    if selected_only:
        _df = _df.loc[:, col_format.keys()]
    _df = _df.reset_index()

    # Use styler api to format the table environment.
    s: Styler = _df.style

    # set default escape on alle columns
    s.format(escape="latex")
    # add specific formatter
    if col_format is not None:
        for col, func in col_format.items():
            s = s.format(formatter=func, subset=col, escape="latex")

    s = s.format_index(escape="latex", axis=1)
    s.set_table_styles(
        [
            {"selector": "toprule", "props": ":toprule"},
            {"selector": "midrule", "props": ":midrule"},
            {"selector": "bottomrule", "props": ":bottomrule"},
        ],
        overwrite=False,
    )
    s = s.hide(axis="index")

    with open(path, "w") as fd:
        fd.write(str_replace(s.to_latex(column_format="c" * _df.shape[1])))


class LazyDataFrame(object):
    """
    Read csv to DataFrame with Metadata.
    First line with '#' at the start of the file
    contains metadata of the form KEY1=VAL1,KEY2=VAL2,...
    """

    @classmethod
    def from_path(cls, path):
        return cls(path)

    def __init__(self, path):
        self.path = path
        self.dtype = {}

    def read_meta_data(self, default=None):
        default = (
            {"IDXCOL": 1, "DATACOL": -1, "SEP": ";"} if default is None else default
        )
        with open(self.path, "r") as f:
            meta_data = f.readline().strip()
        if meta_data.startswith("#"):
            meta_data = meta_data[1:]
            meta_data = {
                i.split("=")[0].strip(): i.split("=")[1].strip()
                for i in meta_data.split(",")
            }
            # replace quoted space with simple space
            if "SEP" in meta_data:
                if meta_data["SEP"] == "' '":
                    meta_data["SEP"] = " "
            else:
                meta_data["SEP"] = ";"
            return meta_data
        else:
            return default

    def as_string(self, remove_meta=False):
        if remove_meta:
            with open(self.path, "r") as f:
                meta = f.readline()
                if not meta.startswith("#"):
                    raise ValueError(f"expected metadata row but got 1: {meta}")
                ret = f.read()
        else:
            with open(self.path, "r") as f:
                ret = f.read()

        return ret

    def df(self, set_index=False, column_selection=None, column_names=None):
        meta = self.read_meta_data()
        df: pd.DataFrame = pd.read_csv(
            filepath_or_buffer=self.path,
            sep=meta["SEP"],
            header=0,
            usecols=column_selection,
            dtype=self.dtype,
            decimal=".",
            index_col=False,
            encoding="utf-8",
            comment="#",
        )
        if set_index and "IDXCOL" in meta:
            nr_row_indices = int(meta["IDXCOL"])
            if 0 < nr_row_indices <= df.shape[1]:
                idx_keys = df.columns[:nr_row_indices]
                df = df.set_index(idx_keys.tolist())

        # rename
        if column_names is not None and len(df.columns) == len(column_names):
            if type(column_names) == list:
                df = df.rename(
                    columns={i: c for i, c in zip(list(df.columns), column_names)}
                )
            elif type(column_names) == dict:
                df = df.rename(columns=column_names)
            else:
                TypeError(f"Expected list or dict got {type(column_names)}")
        return df
