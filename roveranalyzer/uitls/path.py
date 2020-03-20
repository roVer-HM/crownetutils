import glob
import json
import os
import pathlib
import re


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


class RelPath:
    """
    Simple helper to remove absolute paths in analysis files.
    """

    @classmethod
    def from_user_home(cls):
        return cls(str(pathlib.Path.home()))

    @classmethod
    def from_env(cls, env_var):
        if env_var in os.environ:
            return cls(os.environ[env_var])
        else:
            raise KeyError(f"no Variable name '{env_var}' found")

    def __init__(self, base):
        self._base = os.path.abspath(base)
        if not os.path.exists(self._base):
            raise FileNotFoundError(f"given base path does not exist. '{self._base}'")

    def join(self, *paths):
        for p in paths:
            if os.path.isabs(p):
                raise ValueError(
                    f"PathHelper only accepts relative paths. '{p}' is not relative"
                )
        return self.abs_path(*paths)

    def glob(self, *paths, recursive=True):
        p = self.join(*paths)
        return glob.glob(p, recursive=recursive)

    def abs_path(self, *paths):
        return os.path.join(self._base, *paths)

    def get_base(self):
        return self._base

    def extend_base(self, *paths):
        self._base = self.join(*paths)
        return self

    def make_dir(self, *paths, mode=0o777, exist_ok=False):
        p = self.join(*paths)
        os.makedirs(p, mode=mode, exist_ok=exist_ok)
        return p
