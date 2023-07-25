from __future__ import annotations

import os
from os.path import join
from typing import Tuple

from crownetutils.analysis.dpmm.builder import DpmmHdfBuilder
from crownetutils.analysis.dpmm.dpmm import MapType
from crownetutils.omnetpp.scave import CrownetSql
from crownetutils.utils.misc import Project


class AnalysisBase:
    @classmethod
    def builder_from_output_folder(
        cls,
        data_root: str,
        hdf_file="data.h5",
        vec_name="vars_rep_0.vec",
        sca_name="vars_rep_0.sca",
        network_name="World",
        epsg_base=Project.UTM_32N,
        override_hdf: bool = False,
    ) -> Tuple[str, DpmmHdfBuilder, CrownetSql]:
        # todo: try catch here?
        builder = DpmmHdfBuilder.get(
            hdf_file, data_root, override_hdf=override_hdf
        ).epsg(epsg_base)

        sql: CrownetSql = CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
        if sql.is_entropy_map:
            builder.set_map_type(MapType.ENTROPY)
        else:
            builder.set_map_type(MapType.DENSITY)
        return data_root, builder, sql

    @classmethod
    def builder_from_suqc_output(
        cls,
        data_root: str,
        out,
        parameter_id,
        run_id=0,
        hdf_file="data.h5",
        vec_name="vars_rep_0.vec",
        sca_name="vars_rep_0.sca",
        network_name="World",
        epsg_base=Project.UTM_32N,
    ) -> Tuple[str, DpmmHdfBuilder, CrownetSql]:
        data_root = join(
            data_root, "simulation_runs/outputs", f"Sample_{parameter_id}_{run_id}", out
        )
        print(data_root)
        builder = DpmmHdfBuilder.get(hdf_file, data_root).epsg(epsg_base)

        sql = CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
        if sql.is_entropy_map:
            builder.set_map_type(MapType.ENTROPY)
        else:
            builder.set_map_type(MapType.DENSITY)
        return data_root, builder, sql

    @staticmethod
    def find_selection_method(builder: DpmmHdfBuilder):
        p = builder.build().map_p
        return p.get_attribute("used_selection")
