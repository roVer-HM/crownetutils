import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from fs.tempfs import TempFS

from crownetutils.analysis.dpmm.metadata import DpmmMetaData
from crownetutils.analysis.hdf.provider import ProviderVersion


def create_tmp_fs(name, auto_clean=True) -> TempFS:
    tmp_fs = TempFS(identifier=name, auto_clean=auto_clean, ignore_clean_errors=True)
    return tmp_fs


from crownetutils.analysis.dpmm.hdf.dpmm_count_provider import DpmmCountKey
from crownetutils.analysis.dpmm.hdf.dpmm_provider import DpmmKey


def make_dirs(path: Any) -> None:
    if isinstance(path, str):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
    elif isinstance(path, list):
        for p in path:
            make_dirs(p)
    else:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)


def create_count_map_dataframe(number_entries: int = 50) -> pd.DataFrame:
    idxs = [i for i in range(number_entries)]
    xs = [float(i) for i in range(number_entries)]
    ys = [float(i) for i in range(number_entries)]
    ids = [i for i in range(number_entries)]
    entries = np.array([xs, xs, xs, xs]).transpose()
    mult_idx = pd.MultiIndex.from_arrays(
        [idxs, xs, ys, ids],
        names=[DpmmCountKey.SIMTIME, DpmmCountKey.X, DpmmCountKey.Y, DpmmCountKey.ID],
    )
    df = pd.DataFrame(
        entries,
        index=mult_idx,
        columns=[
            DpmmCountKey.COUNT,
            DpmmCountKey.ERR,
            DpmmCountKey.OWNER_DIST,
            DpmmCountKey.SQERR,
        ],
    )
    # additional cases
    df.loc[(42, 42.0, 42.0, 43)] = [42.0, 42.0, 42.0, 43.0]
    df.loc[(42, 42.0, 43.0, 42)] = [42.0, 42.0, 43.0, 42.0]
    df.loc[(42, 42.0, 43.0, 43)] = [42.0, 42.0, 43.0, 43.0]
    df.loc[(42, 43.0, 42.0, 42)] = [42.0, 43.0, 42.0, 42.0]
    df.loc[(42, 43.0, 42.0, 43)] = [42.0, 43.0, 42.0, 43.0]
    df.loc[(42, 43.0, 43.0, 42)] = [42.0, 43.0, 43.0, 42.0]
    df.loc[(42, 43.0, 43.0, 43)] = [42.0, 43.0, 43.0, 43.0]
    return df.sort_index()


def create_dcd_csv_dataframe(
    number_entries: int = 50, node_id: int = 42
) -> Tuple[pd.DataFrame, DpmmMetaData]:
    int_values = [i for i in range(number_entries)]
    float_values = [float(i) for i in range(number_entries)]
    nodes = [node_id for _ in range(number_entries)]
    selections = ["ymf" if i % 2 == 0 else float("nan") for i in range(number_entries)]
    own_cells = [1 for _ in range(number_entries)]

    df = pd.DataFrame(data=None)
    df[DpmmKey.SIMTIME] = int_values
    df[DpmmKey.X] = int_values
    df[DpmmKey.Y] = int_values
    df[DpmmKey.COUNT] = int_values
    df[DpmmKey.MEASURE_TIME] = float_values
    df[DpmmKey.RECEIVED_TIME] = float_values
    df[DpmmKey.SOURCE] = nodes
    df[DpmmKey.SELECTION] = selections
    df[DpmmKey.OWN_CELL] = own_cells
    df.set_index(
        list(DpmmKey.types_csv_index[ProviderVersion.V0_1].keys()),
        drop=True,
        inplace=True,
    )
    return df, DpmmMetaData(3.0, 10, 10, 0)


def safe_dataframe_to_hdf(
    dataframe: pd.DataFrame, hdf_group_key: str, path: str
) -> None:
    dataframe.to_hdf(
        path_or_buf=path, key=hdf_group_key, format="table", data_columns=True
    )
