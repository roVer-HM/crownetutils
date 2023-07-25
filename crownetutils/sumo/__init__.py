import os
import shutil
from dataclasses import dataclass

from crownetutils.utils.logging import logger


@dataclass
class SimDir:
    sim_root: str
    output_root: str = "output"
    route_root: str = "routing_data"
    fcd_root: str = "fcd"
    bm_root: str = "bonnmotion"
    sumo_cfg_root: str = "cfg"

    def __post_init__(self):
        self.sim_root = self._no_trailing_sep(self.sim_root)
        self.output_root = self._no_trailing_sep(self.output_root)
        self.route_root = self._no_trailing_sep(self.route_root)
        self.fcd_root = self._no_trailing_sep(self.fcd_root)
        self.bm_root = self._no_trailing_sep(self.bm_root)
        self.sumo_cfg_root = self._no_trailing_sep(self.sumo_cfg_root)

    def copy_to_clean(self, out_suffix):
        new_sim_dir = SimDir(
            sim_root=self.sim_root,
            output_root=f"{self.output_root}{out_suffix}",
            route_root=self.route_root,
            fcd_root=self.fcd_root,
            bm_root=self.bm_root,
            sumo_cfg_root=self.sumo_cfg_root,
        )
        if os.path.exists(new_sim_dir.out()):
            raise ValueError("New output dir has to be empty")

        new_sim_dir.create_dirs()
        print("copy fixed route files")
        shutil.copytree(
            src=self.route(),
            dst=new_sim_dir.route(),
            dirs_exist_ok=True,
        )
        print("copy sumocfg files")
        shutil.copytree(
            src=self.sumo_cfg(), dst=new_sim_dir.sumo_cfg(), dirs_exist_ok=True
        )
        return new_sim_dir

    def _no_trailing_sep(self, x: str):
        if len(x) > 0 and x[-1] in ["/", "\\"]:
            logger.warn(f"got trailing path separator. Will be removed {x} -> {x[:-1]}")
            x = x[:-1]
        return x

    def create_dirs(self):
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
