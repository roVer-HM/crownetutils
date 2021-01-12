import functools
import glob
import os
import pathlib
import pickle
import re
from enum import Enum


def from_pickle(_func=None, *, path="./analysis.p"):
    """
    Use as decorator for function which return expensive objects to create such as
    pd.DataFrames or objects containing such data.
    If a function is decorated. The function execution is postponed and it is first
    checked if the given path points to a pickle file. If yes the pickle is read and
    returned instead of executing the function.
    """

    def _pickle(func):
        @functools.wraps(func)
        def _wrap(*args, **kwargs):
            if path is not None and os.path.exists(path):
                print(f"read from pickle {path}")
                ret = pickle.load(open(path, "rb"))
            else:
                print(f"crate from raw data {path}")
                ret = func(*args, **kwargs)

            if path is not None and not os.path.exists(path):
                print(f"expected pickle but did not exist.")
                print(f"save pickle now to {path}")
                pickle.dump(ret, open(path, "wb"))
            return ret

        return _wrap

    if _func is None:
        # decorator with arguments
        return _pickle
    else:
        # decorator without arguments
        return _pickle(_func)


class JsonPath:
    """
    return values from a json file represented as a python dict using slashes.

    e.g.
    scenario/topography/sources[1]/shape  --> return shape object of source 1
    scenario/topography/sources[*]/shape  --> return list of all shape objects.
    """

    def __init__(self, path_str):
        p_ele = path_str.split("/")
        self.path = []
        for e in p_ele:
            if "[" in e and "]" in e:
                self.path.append(JsonPathVecEl(e))
            else:
                self.path.append(JsonPathEl(e))

    def _is_last_element(self, idx):
        return idx == len(self.path) - 1

    def get(self, data):
        curr_data = data
        for idx, e in enumerate(self.path):
            curr_data = curr_data[e.key]

            # check if current path element is a vector.
            if type(e) == JsonPathVecEl:
                if e.is_single_item():
                    # select given index and proceed
                    curr_data = curr_data[e.vec_idx]
                else:
                    if self._is_last_element(idx):
                        return curr_data
                    else:
                        vec_data = []
                        for vec_element in curr_data:
                            r = JsonPath(self.sub_path_str(idx + 1)).get(vec_element)
                            if type(r) == list:
                                vec_data.extend(r)
                            else:
                                vec_data.append(r)
                        return vec_data
                        # raise ValueError(f"*-Syntax in the middle of a path not supported.")
        return curr_data

    def sub_path_str(self, f):
        return "/".join([e.path_str for e in self.path[f:]])

    def __str__(self):
        return "/".join([e.key for e in self.path])


class JsonPathEl:
    def __init__(self, path=""):
        self.path = path

    def is_vec(self):
        return False

    @property
    def key(self):
        return self.path

    @property
    def path_str(self):
        return self.key


class JsonPathVecEl(JsonPathEl):
    group_matcher = "^(.*)\[(.*)\]$"

    def __init__(self, path):
        parts = re.fullmatch(self.group_matcher, path)
        if parts is None:
            raise ValueError(f"expected a vector item got {path}")
        vec = parts.groups()[1]
        if vec != "*":
            try:
                vec = int(vec)
            except ValueError:
                raise ValueError(f"expected '*' or int as index got '{vec}' ")
        self.vec_idx = vec
        super().__init__(parts.groups()[0])

    def is_single_item(self):
        return type(self.vec_idx) == int

    def is_vec(self):
        return True

    @property
    def path_str(self):
        return f"{self.key}[{self.vec_idx}]"


class Suffix(Enum):
    HDF = ".h5"
    CSV = ".csv"
    PNG = ".png"
    PDF = ".pdf"


class PathHelper:
    """
    Simple helper to remove absolute paths in analysis files.
    """

    @classmethod
    def rover_sim(cls, sim_name, *extend_base):
        return cls.from_env(
            "CROWNET_HOME", "rover/simulations/", sim_name, "results", *extend_base
        )

    @classmethod
    def from_user_home(cls):
        return cls(str(pathlib.Path.home()))

    @classmethod
    def from_env(cls, env_var, *extend_base):
        if env_var in os.environ:
            c = cls(os.environ[env_var])
            if len(extend_base) > 0:
                c.extend_base(*extend_base)
            return c
        else:
            raise KeyError(f"no Variable name '{env_var}' found")

    def __init__(self, base, create_missing=False):
        self._base = os.path.abspath(base)
        if create_missing and not os.path.exists(self._base):
            os.makedirs(self._base, exist_ok=True)
        if not os.path.exists(self._base):
            raise FileNotFoundError(f"given base path does not exist. '{self._base}'")

    def join(self, *paths):
        for p in paths:
            if os.path.isabs(p):
                raise ValueError(
                    f"PathHelper only accepts relative paths. '{p}' is not relative"
                )
        return self.abs_path(*paths)

    def glob(self, *paths, recursive=True, expect=-1):
        p = self.join(*paths)
        _ret = glob.glob(p, recursive=recursive)
        if expect <= 0:
            # expect any output empyt list, 1, 2,... items
            return _ret
        elif len(_ret) == expect:
            if len(_ret) == 1:
                return _ret[0]
            else:
                return _ret
        else:
            raise ValueError(f"expected {expect} items but got {len(_ret)}")

    def abs_path(self, *paths):
        return os.path.join(self._base, *paths)

    def get_base(self):
        return self._base

    def extend_base(self, *paths):
        self._base = self.join(*paths)
        return self

    def make_dir(self, *paths, mode=0o777, exist_ok=False):
        d_path = self.join(*paths)
        os.makedirs(d_path, exist_ok=exist_ok)
        return d_path


class ResultPath(PathHelper):
    @classmethod
    def create(cls, simulation, run_name):
        _obj = cls.rover_sim(simulation, run_name)
        return _obj

    @property
    def scenario_path(self):
        _p = self.glob("vadere.d/*.scenario")
        if len(_p) == 0:
            raise FileNotFoundError(
                f"Expected scenario file at {self.abs_path('vadere.d')}"
            )
        elif len(_p) > 1:
            print(
                f"Warning: expected one scenario file but found more: [{' '.join(_p)}]"
            )

        return _p[0]
