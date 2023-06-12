import os
from dataclasses import dataclass


@dataclass
class SimDir:
    sim_root: str
    output_root: str = "output"
    route_root: str = "routing_data"
    fcd_root: str = "fcd"
    bm_root: str = "bonnmotion"
    sumo_cfg_root: str = "cfg"

    def __post_init__(self):
        os.makedirs(self.out(ensure_dir_exists=True), exist_ok=True)
        os.makedirs(self.route(ensure_dir_exists=True), exist_ok=True)
        os.makedirs(self.fcd(ensure_dir_exists=True), exist_ok=True)
        os.makedirs(self.bm(ensure_dir_exists=True), exist_ok=True)
        os.makedirs(self.sumo_cfg(ensure_dir_exists=True), exist_ok=True)

    def root(self, *args):
        return os.path.join(self.sim_root, *args)

    def _join(self, ensure_dir_exists: bool = False, *args):
        p = os.path.join(*args)
        if ensure_dir_exists and not os.path.exists(os.path.dirname(p)):
            os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def out(self, *args, ensure_dir_exists: bool = False):
        if os.path.isabs(self.output_root):
            return self._join(ensure_dir_exists, self.output_root, *args)
        else:
            return self._join(ensure_dir_exists, self.sim_root, self.output_root, *args)

    def _path(self, _dir, ensure_dir_exists: bool = False, *args):
        if os.path.isabs(_dir):
            return self._join(ensure_dir_exists, _dir, *args)
        else:
            return self._join(ensure_dir_exists, self.out(), _dir, *args)

    def route(self, *args, ensure_dir_exists: bool = False):
        return self._path(self.route_root, ensure_dir_exists, *args)

    def fcd(self, *args, ensure_dir_exists: bool = False):
        return self._path(self.fcd_root, ensure_dir_exists, *args)

    def bm(self, *args, ensure_dir_exists: bool = False):
        return self._path(self.bm_root, ensure_dir_exists, *args)

    def sumo_cfg(self, *args, ensure_dir_exists: bool = False):
        return self._path(self.sumo_cfg_root, ensure_dir_exists, *args)
