import contextlib
import glob
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from roveranalyzer.simulators.opp.configuration import Config
from roveranalyzer.simulators.opp.scave import ScaveRunConverter, ScaveTool
from roveranalyzer.simulators.vadere.scenario_output import ScenarioOutput
from roveranalyzer.utils import PathHelper


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
        warnings.warn("hdf rework", DeprecationWarning)
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

    def __init__(
        self,
        path: PathHelper,
        analysis_name,
        analysis_dir=None,
        hdf_store_name=None,
        cfg: Config = None,
    ):
        warnings.warn("hdf rework", DeprecationWarning)
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
        self._hdf_store = HdfProvider(
            self._root.join(self._analysis_dir, _hdf_store_name)
        )

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
