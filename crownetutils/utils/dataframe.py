from __future__ import annotations

from functools import partial
from glob import escape
from typing import Any, Callable, List, Literal, Protocol

import numpy as np
import pandas as pd
from pandas._typing import IntervalClosedType
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
    """_summary_
    """

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


def numeric_formatter(
    cmd="num", precision: int = 4, *args, **kwargs
) -> Callable[[Any], str]:
    """Wrapper function to apply formatting to numeric values only"""
    _si = siunitx(cmd, precision, *args, **kwargs)

    def _f(val, *args, **kwargs):
        if any([isinstance(val, _i) for _i in [float, int]]):
            return _si(val)
        else:
            return str(val)

    return _f


def format_frame(
    df: pd.DataFrame, si_func=lambda x: f"\\num{{{x}}}", col_list=None
) -> pd.DataFrame:
    _df: pd.DataFrame = df.copy(deep=True)
    if isinstance(si_func, dict):
        for col, _func in si_func.items():
            if col in _df.columns:
                _df[col] = _df[col].apply(_func)
    elif col_list is not None:
        for col in col_list:
            _df[col] = _df[col].apply(si_func)
    else:
        _df = _df.applymap(si_func)

    return _df


def save_as_tex_table(
    df: pd.DataFrame,
    path: str | None = None,
    selected_only: bool = False,
    rename: dict | None = None,
    col_format: dict | List | None = None,
    add_default_latex_format: bool = True,
    str_replace: Callable[[str], str] = lambda x: x,
):
    _df: pd.DataFrame = df.copy(deep=True)
    if rename is not None:
        _df = _df.rename(columns=rename)
    if selected_only and col_format is not None:
        if isinstance(col_format, dict):
            _df = _df[col_format.keys()]
        else:
            c = [col[0] for col in col_format]
            _df = _df[c]

    # Use styler api to format the table environment.
    s: Styler = _df.style

    if add_default_latex_format:
        # set default escape on alle columns
        s.format(escape="latex")
    # add specific formatter
    if col_format is not None:
        if isinstance(col_format, dict):
            for col, func in col_format.items():
                s = s.format(formatter=func, subset=col, escape="latex")
        elif isinstance(col_format, list):
            for col, func in col_format:
                s = s.format(formatter=func, subset=col, escape="latex")
        else:
            s = s.format(formatter=col_format, subset=df.columns, escape="latex")

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

    if path is None:
        return str_replace(s.to_latex(column_format="c" * _df.shape[1]))
    else:
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


def append_index(df: pd.DataFrame, col: str, val=None):
    if col not in df.columns and val is not None:
        df[col] = val
    _idx = [*df.index.names, col]
    df = df.reset_index().set_index(_idx)
    return df


def append_columns(
    df: pd.DataFrame, column_names: str | List[str], default_value: Any | List[Any]
) -> pd.DataFrame:
    """Append columns to data frame with the provided default values.
    If both column_names and default_value are lists, there length must match. If the default_values is not a list
    the value will be used by each column

    Args:
        df (pd.DataFrame): Frame which will be extended
        column_names (str | List[str]): column names to add to frame
        default_value (Any | List[Any]): default values used.
    Raises:
        ValueError:

    Returns:
        pd.DataFrame:
    """

    column_names = [column_names] if isinstance(column_names, str) else column_names
    if isinstance(default_value, list):
        if len(default_value) != len(column_names):
            raise ValueError(
                f"column_names and default values must match length. got {len(column_names)} != {len(default_value)}"
            )
    else:
        default_value = np.repeat([default_value], len(column_names))
    for idx, col in enumerate(column_names):
        df[col] = default_value[idx]
    return df


def index_or_col(df, name):
    if isinstance(df.index, pd.MultiIndex) and name in df.index.names:
        return df.index.get_level_values(name)
    elif df.index.name == name:
        return df.index
    elif isinstance(df, pd.Series) and name in df.index:
        # frame with one row  is reduced to pd.Series with columns as index.
        # return as list with one item
        return [df[name]]
    elif name in df.columns:
        return df[name]
    else:
        raise ValueError(f"name {name} not found in index or columns")


def append_interval(
    frame: pd.DataFrame,
    interval_range: float,
    time_col: str = "time",
    start_time: float = 0.0,
    end_time: float | None = None,
    closed: IntervalClosedType = "left",
    interval_col: str | None = "time_bin",
):
    time_data = index_or_col(frame, time_col)
    end_time = (
        np.ceil(time_data.max() + interval_range) if end_time is None else end_time
    )
    bins = pd.interval_range(
        start=start_time, end=end_time, freq=interval_range, closed=closed
    )
    if interval_col is None:
        return pd.cut(time_data, bins)
    else:
        frame[interval_col] = pd.cut(time_data, bins)
        return frame


def build_interval(frame: pd.DataFrame, provider) -> pd.DataFrame:
    m = provider.get_attribute("interval_column")
    if all(i in frame.columns for i in [m[1], m[2]]):
        idx = [
            pd.Interval(v[0], v[1], closed=m[3])
            for v in frame.loc[:, [m[1], m[2]]].values
        ]
        frame[m[0]] = pd.IntervalIndex(idx, closed=m[3])
    return frame


def get_index_name_or_names(df):
    if isinstance(df.index, pd.MultiIndex):
        return df.index.names
    else:
        return df.index.name


def merge_on_interval(
    data: pd.DataFrame,
    df_interval,
    index="time",
    interval_col="interval",
    interval_closed_at: IntervalClosedType = "left",
    merge: bool = True,
    copy_data: bool = True,
    **merge_args,
):
    if copy_data:
        data = data.copy()
    if isinstance(data.index, pd.MultiIndex):
        group_by_index = list(data.index.names)
        if index not in group_by_index:
            raise ValueError(
                f"Expected  index level {index} in dataframe. Got '{group_by_index}'"
            )
        group_by_index.remove(index)

        data[interval_col] = np.nan
        for _index, _df in data.groupby(group_by_index):
            if isinstance(_index, tuple) and len(_index) == 1:
                _index = _index[0]
            i_index = pd.IntervalIndex(
                index_or_col(df_interval.loc[_index,], interval_col),
                closed=interval_closed_at,
            )
            data.loc[_index, [interval_col]] = pd.cut(
                _df.index.get_level_values(index), bins=i_index
            )
    else:
        i_index = pd.IntervalIndex(
            index_or_col(df_interval, interval_col), closed=interval_closed_at
        )

        data[interval_col] = pd.cut(data.index.get_level_values(0), bins=i_index)
        group_by_index = []

    if merge:
        merge_args.update(dict(on=[*group_by_index, interval_col], how="left"))
        out = data.reset_index().merge(df_interval.reset_index(), **merge_args)

    return out


def partial_index_match(df: pd.DataFrame, partial_idx: pd.MultiIndex) -> pd.DataFrame:
    # save old index to recreate later
    idx_old = df.index.names
    # drop parts that do not match partial index
    idx_drop = [i for i in df.index.names if i not in partial_idx.names]
    df = df.reset_index(idx_drop)
    if df.index.names != partial_idx.names:
        raise ValueError("Index mismatch")
    # select items and recreated index
    idx = partial_idx.intersection(df.index)
    df = df.loc[idx]
    df = df.reset_index().set_index(idx_old)
    return df


class DataFrameStructureError(Exception):
    pass


def assert_frame_structure(
    df: pd.DataFrame, index_names=None, column_names=None, shape=(-1, -1), msg: str = ""
):
    if index_names is not None:
        if isinstance(index_names, str):
            if df.index.name != index_names:
                raise DataFrameStructureError(
                    f"expected index name '{index_names}' got '{df.index.name}' msg: {msg}"
                )
        else:
            if (len(df.index.names) != len(index_names)) or any(
                [i1 != i2 for (i1, i2) in zip(df.index.names, index_names)]
            ):
                raise DataFrameStructureError(
                    f"expected index  '{index_names}' got '{df.index.names}'"
                )
    if column_names is not None:
        if isinstance(column_names, str):
            column_names = [column_names]
        if len(df.columns) != len(column_names) or any(
            [i1 != i2 for (i1, i2) in zip(list(df.columns.values), column_names)]
        ):
            raise DataFrameStructureError(
                f"expected columns '{column_names}' got '{list(df.columns.values)}' msg: {msg}"
            )
