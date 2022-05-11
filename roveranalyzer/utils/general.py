import os
from cgitb import reset
from functools import partial
from multiprocessing.managers import ValueProxy

import pyproj
from shapely.geometry import geo
from shapely.ops import transform
from tenacity import retry


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
