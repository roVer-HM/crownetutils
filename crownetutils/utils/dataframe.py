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


class MissingValueImputationStrategy(Protocol):
    """Imputation strategy to fill or remove missing values from a frame. Note that in case of removing the whole
    row will be removed."""

    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        ...


class ArbitraryValueImputation(MissingValueImputationStrategy):
    def __init__(self, fill_value=0.0) -> None:
        self.fill_value = fill_value

    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        df[data_column] = df[data_column].fillna(self.fill_value)
        return df


class DeleteMissingImputation(MissingValueImputationStrategy):
    def __call__(
        self, df: pd.DataFrame, data_column, *args: Any, **kwds: Any
    ) -> pd.DataFrame:
        mask = ~df[data_column].isna()
        return df[mask].copy()


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
