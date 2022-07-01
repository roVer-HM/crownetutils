from __future__ import annotations

from typing import Protocol

import pandas as pd


class FrameConsumer(Protocol):

    # the 'do nothing consumer'
    EMPTY: FrameConsumer = lambda df: df

    def __call__(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        pass


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
