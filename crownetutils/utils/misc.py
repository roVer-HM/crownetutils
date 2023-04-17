import contextlib as c
import locale
import os
import time
from functools import partial
from typing import Any, List

import numpy as np
import pyproj
from shapely.ops import transform


class Result:
    @classmethod
    def ok(cls, result=None):
        return cls(ret_code=True, result=result)

    @classmethod
    def err(cls, msg):
        return cls(ret_code=False, result=None, msg=msg)

    def __init__(self, ret_code: bool, result: Any, msg: str = "") -> None:
        self.ret_code = ret_code
        self.result = result
        self.msg = msg

    def get(self):
        return self.result

    def __bool__(self):
        return self.ret_code

    def __repr__(self) -> str:
        return f"<Result: {self.ret_code}, msg='{self.msg}'"


@c.contextmanager
def change_locale(category=locale.LC_ALL, loc="de_DE.utf8"):
    try:
        old = locale.getlocale()
        locale.setlocale(category, loc)
        yield
    finally:
        locale.setlocale(category, old)


def ccw(a, b, c):
    """
    is triangle abc counter clockwise?
    """
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(line1, line2):
    assert line1.shape == line2.shape == (2, 2)
    return ccw(line1[0], line2[0], line2[1]) != ccw(
        line1[1], line2[0], line2[1]
    ) and ccw(line1[0], line1[1], line2[0]) != ccw(line1[0], line1[1], line2[1])


def apply_str_filter(filter: str, data: List[int]):
    data = list(data)
    data.sort()
    groups = filter.split(",")
    ranges = [g.split("-") for g in groups]
    output = []
    try:
        for r in ranges:
            if len(r) == 1:
                val = int(r[0])
                if val in data:
                    output.append(val)
            elif len(r) == 2:
                if r[1] == "":
                    r[1] = max(data)
                left, right = [int(i) for i in r]
                if left > right:
                    raise ValueError("")
                for v in data:
                    if v >= left and v <= right:
                        output.append(v)
    except ValueError:
        raise ValueError(f"cannot parse filter {filter}")

    return output


class ProgressCmd:
    def __init__(self, cycle_count, prefix="", print_interval=0.05):
        self.cycle_count = cycle_count
        self.print_interval = print_interval
        self.prefix = prefix
        self.curr = 0
        self.curr_th = 0.0

    def incr(self, cycle=1):
        self.curr += cycle
        if (self.curr / self.cycle_count) > self.curr_th:
            self.curr_th += self.print_interval
            print(
                f"\r{self.prefix}{self.curr:03}/{self.cycle_count:03}--{(self.curr / self.cycle_count):02.2%}",
                end="",
            )
        if self.curr >= self.cycle_count:
            print("")


class Timer:
    ACTIVE = True

    @classmethod
    def create_and_start(cls, name="timer", label=0):
        return cls(name, label)

    def __init__(self, name, label):
        self._name = name
        self._label = label
        self._start = time.time()

    def stop(self):
        if self.ACTIVE:
            print(
                f"{self._label}::timer>> {time.time() - self._start:0.5f}s\t({self._name})"
            )
        return self

    def start(self, name):
        self._name = name
        self._start = time.time()
        return self

    def stop_start(self, new_name):
        self.stop()
        self.start(new_name)
        return self


class StatsTool:
    """
    Toolset for calculating and nicely printing statistics
    """

    @staticmethod
    def stats_table(data, unit: str = "", name: str = "") -> str:
        """
        Create a table listing the most important statistic values

        :param data:    data to calculate the statistics on
        :param unit:    SI unit of data (optional)
        :param name:    name of the data to be printed (optional)
        :return:        string with statistics table
        """
        table = "=============================================================\n"
        if len(name) > 0:
            table += (
                f"! Data: {name:51} !\n"
                "-------------------------------------------------------------\n"
            )

        table += (
            f"! nr of values : {len(data):15}                            !\n"
            f"! arith. mean  : {np.mean(data):15.6f} {unit:>4}                       !\n"
            f"! minimum      : {np.min(data):15.6f} {unit:>4}                       !\n"
            f"! maximum      : {np.max(data):15.6f} {unit:>4}                       !\n"
            f"! median       : {np.median(data):15.6f} {unit:>4}                       !\n"
            f"! std. dev.    : {np.std(data):15.6f} {unit:>4}                       !\n"
            f"! variance     : {np.var(data):15.6f} {unit:>4}^2                     !\n"
            "=============================================================\n"
        )

        return table


def add_rover_env_var():
    os.environ["CROWNET_HOME"] = os.path.abspath("../../../")
    if os.environ["CROWNET_HOME"] is None:
        raise SystemError(
            "Please add CROWNET_HOME to your system variables to run a rover simulation."
        )


class DataSource:
    @classmethod
    def provide_result(cls, name, source, result):
        c = cls(name, source)
        c.ret = result
        return c

    def __init__(self, name, source) -> None:
        self.name = name
        self.ret = None
        self.source = source

    def __repr__(self) -> str:
        return f"<{self.__name__}: {self.name} source:{self.source}>"

    def __call__(self, *args, **kwargs):
        return self.ret


class Project:
    EPSG4326 = "EPSG:4326"
    EPSG3857 = "EPSG:3857"
    EPSG32632 = "EPSG:32632"
    UTM_32N = EPSG32632
    WSG84_lat_lon = EPSG4326
    WSG84_pseudo_mercator = EPSG3857
    OpenStreetMaps = WSG84_pseudo_mercator
    GoogleMaps = WSG84_pseudo_mercator

    @classmethod
    def fromLatLon(cls, crs=WSG84_lat_lon):
        return cls(source_crs=crs)

    @classmethod
    def fromOSM(cls):
        return cls(source_crs=cls.OpenStreetMaps)

    @classmethod
    def from_proj(cls, pro_str):
        if pro_str == "+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs":
            return cls.UTM_32N
        else:
            raise ValueError("not supported")

    def __init__(self, source_crs=None, dest_crs=None) -> None:
        self.source_crs = source_crs
        self.dest_crs = dest_crs
        self._project = None
        self._check_projection()

    def _check_projection(self):
        if (
            self._project is None
            and self.source_crs is not None
            and self.dest_crs is not None
        ):
            self._project = partial(
                pyproj.transform,
                pyproj.Proj(self.source_crs),
                pyproj.Proj(self.dest_crs),
            )

    def to(self, crs):
        self.dest_crs = crs
        self._project = None
        return self

    def to_osm(self):
        return self.to(crs=self.OpenStreetMaps)

    def transfrom(self, geom):
        self._check_projection()
        if type(geom) == list:
            return [transform(self._project, g) for g in geom]
        else:
            return transform(self._project, geom)
