import json
import os
import pickle

from roveranalyzer.simulators.crownet.dcd.dcd_map import DcdMap2D, DcdMap2DMulti
from roveranalyzer.simulators.crownet.dcd.util import *
from roveranalyzer.simulators.vadere.plots.scenario import VaderScenarioPlotHelper


class PickleState:
    DEACTIVATED = 0  # do not use pickle
    CSV_ONLY = 1  # only dataframe of given csv files (no location_df, no id mapping, no merging, no features)
    MERGED = 2  # merged frame, location_df, id maping done (no features)
    FULL = 3  # all data needed to build DcDMap


class DcdBuilder:
    VIEW_MAP = 1
    FULL_MAP = 2

    tsc_id_idx_name = "ID"
    tsc_time_idx_name = "simtime"
    tsc_x_idx_name = "x"
    tsc_y_idx_name = "y"
    tsc_source_name = "source"

    VIEW_IDX = [
        tsc_id_idx_name,
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
    ]

    FULL_IDX = [
        tsc_id_idx_name,
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
        tsc_source_name,
    ]

    GLOBAL_IDX = [
        tsc_time_idx_name,
        tsc_x_idx_name,
        tsc_y_idx_name,
    ]

    GLOBAL_COLS = {
        "count": int,
        "node_id": np.str,
    }

    VIEW_COLS = {
        "count": int,
        "measured_t": float,
        "received_t": float,
        "source": np.str,
        "own_cell": int,
    }

    FULL_COLS = {
        "count": int,
        "measured_t": float,
        "received_t": float,
        "source": np.str,
        "own_cell": int,
        "selection": np.str,  # needed to filter out view form other data
    }

    def __init__(self):
        self._map_type = self.VIEW_MAP
        self._map_idx = self.VIEW_IDX
        self._glb_idx = self.GLOBAL_IDX
        self._glb_cols = self.GLOBAL_COLS
        self._map_cols = self.VIEW_COLS
        self._real_coords = True
        self._single_df_filters = []
        self._scenario_plotter = None
        self._features = [delay_feature, owner_dist_feature]
        self._global_csv_path = None
        self._node_csv_paths = []
        self._data_base_path = None
        self._root_pickle = None
        self._pickle_state = PickleState.DEACTIVATED

    def view(self):
        self._map_type = self.VIEW_MAP
        self._map_idx = self.VIEW_IDX
        return self

    def all(self):
        self._map_type = self.FULL_MAP
        self._map_idx = self.FULL_IDX
        self._map_cols = self.FULL_COLS
        return self

    def csv_paths(self, global_csv_path, node_csv_paths):
        self._global_csv_path = global_csv_path
        self._node_csv_paths = node_csv_paths
        return self

    def data_base_path(self, path):
        self._data_base_path = path
        return self

    def pickle_name(self, pickle_name):
        if os.path.isabs(pickle_name):
            self._root_pickle = pickle_name
        else:
            self._root_pickle = os.path.join(self._data_base_path, pickle_name)
        return self

    def pickle_as(self, state: PickleState):
        self._pickle_state = state
        return self

    def data_idx(self, *kw):
        self._map_idx = kw
        return self

    def glb_idx(self, *kw):
        self._glb_idx = kw
        return self

    def glb_cols(self, **kwargs):
        self._glb_cols = kwargs
        return self

    def map_cols(self, **kwargs):
        self._map_cols = kwargs
        return self

    def add_single_filter(self, *kwarg):
        self._single_df_filters.extend(*kwarg)
        return self

    def clear_single_filter(self):
        self._single_df_filters.clear()
        return self

    def clear_features(self):
        self._features.clear()
        return self

    def add_feature(self, *kw):
        self._features.extend(*kw)
        return self

    def use_real_coords(self, val=True):
        self._real_coords = val
        return self

    def plotter(self, plotter):
        if type(plotter) == str:
            self._scenario_plotter = VaderScenarioPlotHelper(plotter)
        else:
            self._scenario_plotter = plotter
        return self

    def build(self):
        if self._pickle_state == PickleState.DEACTIVATED or not os.path.exists(
            self._root_pickle
        ):
            # no pickle path provided build from csv data (may take some time!)
            return self.build_from_csv()

        # read pickle and extract data. Recreate DcdBuilder object used to create the data and check that the pickle is
        # usable for the current DcdBuilder configuration
        _pickle = self.read_pickle()
        _builder = self.from_json(_pickle["builder_json"])
        _state = _builder._pickle_state
        _data = _pickle["data"]
        if not self._compare_builder(_builder):
            raise RuntimeError("pickle not compatible with current builder")

        # Apply missing build steps based on pickle state (if any)
        print(f"found pickle state {_state} build object ...")
        if (
            _state == PickleState.CSV_ONLY
        ):  # no location table, features only csv, (delay gets calulated afterwards)
            # location_df, no id mapping, no merging, no features
            _data = self.do_merge(**_data)
            _data = self.do_feature(**_data)
        elif _state == PickleState.MERGED:
            _data = self.do_feature(**_data)
            _data = self.do_extract_view(**_data)
        elif _state == PickleState.FULL:
            pass  # do nothing
        else:
            raise RuntimeError("unknown PickleState")

        # Create Dcd object (finally...)
        return self._build(
            _data,
            self._map_type,
            plotter=self._scenario_plotter,
            data_base_dir=self._data_base_path,
        )

    def _compare_builder(self, other):
        # other only contains static content! class/method references were striped due to pickle
        return self._pickle_state == other._pickle_state

    def build_from_csv(self):
        data = self.dcd_data_from_csv(self._global_csv_path, self._node_csv_paths)
        return self._build(
            data,
            self._map_type,
            plotter=self._scenario_plotter,
            data_base_dir=self._data_base_path,
        )

    @classmethod
    def _build(cls, data, map_type, **kwargs):
        if map_type == cls.FULL_MAP:
            return DcdMap2DMulti(**data, **kwargs)
        else:
            return DcdMap2D(**data, **kwargs)

    def write_pickle(self, data: dict, **kwargs):
        """
        write given data to pickle file. Add DcdBuilder json and any additional data (kwargs)
        to the pickle. Write json representation of DcdBuilder object with striped references
        to ensure pickles will work with later version of the analysis software. Only pickle
        stable data (DataFrames, Series, Json strings, serializable content)
        """
        if self._root_pickle is None:
            raise RuntimeError("Path not set")
        p = {
            "builder_json": self.to_json(clear_references=True),
            "data": data,
        }
        p.update(kwargs)
        with open(self._root_pickle, "wb") as fd:
            pickle.dump(p, fd)

    def read_pickle(self):
        if self._root_pickle is None:
            raise RuntimeError("Path not set")
        with open(self._root_pickle, "rb") as fd:
            data = pickle.load(fd)
        return data

    def _pickle_if_needed(self, data, state):
        if (
            self._pickle_state != PickleState.DEACTIVATED
            and self._pickle_state == state
        ):
            self.write_pickle(data)

    def dcd_data_from_csv(self, global_data_path, node_data_paths):
        _ret = self.do_csv(global_data_path, node_data_paths)
        self._pickle_if_needed(_ret, PickleState.CSV_ONLY)
        _ret = self.do_merge(**_ret)
        self._pickle_if_needed(_ret, PickleState.MERGED)
        _ret = self.do_feature(**_ret)
        _ret = self.do_extract_view(**_ret)
        self._pickle_if_needed(_ret, PickleState.FULL)
        return _ret

    def do_csv(self, global_data_path, node_data_paths):
        df_global = build_density_map(
            csv_path=global_data_path,
            index=self._glb_idx,
            column_types=self._glb_cols,
            real_coords=self._real_coords,
            df_filter=None,
        )

        # load map for each node *.csv -> list of DataFrames
        njobs = int(multiprocessing.cpu_count() * 0.60)
        with multiprocessing.Pool(processes=njobs) as pool:
            job_args = [
                {
                    "csv_path": p,
                    "index": self.VIEW_IDX[
                        1:
                    ],  # ID will be set later (use always VIEW_IDX here. IN full map source will be set later)
                    "column_types": self._map_cols,
                    "real_coords": self._real_coords,
                    "df_filter": self._single_df_filters,
                }
                for p in node_data_paths
            ]
            df_data = run_pool(pool, build_density_map, job_args)
        return {"global_data": df_global, "node_data": df_data}

    def do_merge(self, global_data, node_data):
        meta_data, global_df = global_data

        # create location df [x, y] -> nodeId (long string)
        print("create location df [x, y] -> nodeId")
        location_df, global_df = self.build_location_df(global_df)

        # ID mapping
        print("create node id mapping")
        location_df, node_id_map = self.build_id_map(location_df, node_data)

        # merge node frames and set index
        print("merge dataframes")
        map_df = self.merge_frames(node_data, node_id_map, self._map_idx)

        _ret = {
            "glb_meta": meta_data,
            "node_id_map": node_id_map,
            "global_df": global_df,
            "map_df": map_df,  # may be full map based on self._type!
            "location_df": location_df,
        }
        return _ret

    def do_extract_view(self, **data):
        if self._map_type == self.FULL_MAP:
            data.setdefault("map_all_df", data["map_df"])
            _map_df = data["map_df"]
            _map_df = (
                _map_df[_map_df["selection"].notnull()]
                .drop(columns=["selection"])
                .reset_index(self._map_idx[-1], drop=False)
            )
            if not _map_df.index.is_unique:
                raise ValueError("map_view_df must have unique index")
            data["map_df"] = _map_df

        return data

    def do_feature(self, **data):
        print("calculate features")
        for _feature in self._features:
            data["map_df"] = _feature(data["map_df"], **data)

        return data

    @staticmethod
    def build_location_df(glb_df):
        # global position map for all node_ids
        glb_loc_df = glb_df["node_id"].copy().reset_index()
        glb_loc_df = glb_loc_df.assign(
            node_id=glb_loc_df["node_id"].str.split(r",\s*")
        ).explode("node_id")
        # remove node_id column from global
        glb_df = glb_df.drop(labels=["node_id"], axis="columns")
        return glb_loc_df, glb_df

    @staticmethod
    def build_id_map(location_df, node_data):
        # read id from meta data object of each node_data tuple
        ids = [n[0].node_id for n in node_data]
        ids.sort()
        ids = list(enumerate(ids, 1))
        ids.insert(0, (0, "-1"))
        node_to_id = {n[1]: n[0] for n in ids}

        location_df["node_id"] = location_df["node_id"].apply(
            lambda x: node_to_id.get(x, -1)
        )
        location_df = location_df.set_index(["simtime", "node_id"])
        return location_df, node_to_id

    @staticmethod
    def merge_frames(input_df, node_to_id, index):
        node_dfs = []
        for meta, _df in input_df:
            if meta.node_id not in node_to_id:
                raise ValueError(f"{meta.node_id} not found in id_map")
            # add ID as column
            _df["ID"] = node_to_id[meta.node_id]
            # replace source string id with int based ID
            _df["source"] = _df["source"].map(node_to_id)
            # remove index an collect in list for later pd.concat
            _df = _df.reset_index()
            node_dfs.append(_df)

        _df_ret = pd.concat(node_dfs, levels=index, axis=0)
        _df_ret = _df_ret.set_index(index)
        _df_ret = _df_ret.sort_index()
        return _df_ret

    def to_json(self, clear_references=True):
        _obj = dict(self.__dict__)
        if clear_references:
            replace_default = {
                "_single_df_filters": [],
                "_scenario_plotter": None,
                "_features": [],
            }
            for key, value in replace_default.items():
                _obj[key] = value
        return json.dumps(_obj, default=lambda o: str(o), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, json_str):
        _json = json.loads(json_str)
        _obj = cls()
        for key, val in _json.items():
            setattr(_obj, key, val)

        return _obj
