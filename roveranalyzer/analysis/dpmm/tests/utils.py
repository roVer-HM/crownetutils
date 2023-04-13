import os
from typing import Any, Tuple

import numpy as np
import pandas as pd
from fs.tempfs import TempFS

from roveranalyzer.analysis.hdfprovider.IHdfProvider import ProviderVersion
from roveranalyzer.simulators.crownet.common.dcd_metadata import DcdMetaData


def create_tmp_fs(name, auto_clean=True) -> TempFS:
    tmp_fs = TempFS(identifier=name, auto_clean=auto_clean, ignore_clean_errors=True)
    return tmp_fs


from roveranalyzer.analysis.dpmm.DcdMapCountProvider import CountMapKey
from roveranalyzer.analysis.dpmm.DcdMapProvider import DcdMapKey


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
        names=[CountMapKey.SIMTIME, CountMapKey.X, CountMapKey.Y, CountMapKey.ID],
    )
    df = pd.DataFrame(
        entries,
        index=mult_idx,
        columns=[
            CountMapKey.COUNT,
            CountMapKey.ERR,
            CountMapKey.OWNER_DIST,
            CountMapKey.SQERR,
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
) -> Tuple[pd.DataFrame, DcdMetaData]:
    int_values = [i for i in range(number_entries)]
    float_values = [float(i) for i in range(number_entries)]
    nodes = [node_id for _ in range(number_entries)]
    selections = ["ymf" if i % 2 == 0 else float("nan") for i in range(number_entries)]
    own_cells = [1 for _ in range(number_entries)]

    df = pd.DataFrame(data=None)
    df[DcdMapKey.SIMTIME] = int_values
    df[DcdMapKey.X] = int_values
    df[DcdMapKey.Y] = int_values
    df[DcdMapKey.COUNT] = int_values
    df[DcdMapKey.MEASURE_TIME] = float_values
    df[DcdMapKey.RECEIVED_TIME] = float_values
    df[DcdMapKey.SOURCE] = nodes
    df[DcdMapKey.SELECTION] = selections
    df[DcdMapKey.OWN_CELL] = own_cells
    df.set_index(
        list(DcdMapKey.types_csv_index[ProviderVersion.V0_1].keys()),
        drop=True,
        inplace=True,
    )
    return df, DcdMetaData(3.0, 10, 10, 0)


def safe_dataframe_to_hdf(
    dataframe: pd.DataFrame, hdf_group_key: str, path: str
) -> None:
    dataframe.to_hdf(
        path_or_buf=path, key=hdf_group_key, format="table", data_columns=True
    )
