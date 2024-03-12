from __future__ import annotations

import os
from os.path import join
from typing import Tuple

from crownetutils.analysis.dpmm import MapType
from crownetutils.analysis.dpmm.builder import DpmmHdfBuilder
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg, DpmmCfgCsv
from crownetutils.omnetpp.scave import CrownetSql
from crownetutils.utils.logging import logger
from crownetutils.utils.misc import Project


class AnalysisBase:
    @classmethod
    def builder_from_cfg(cls, cfg: DpmmCfg, override_hdf: bool = False):
        builder = DpmmHdfBuilder.get(cfg=cfg, override_hdf=override_hdf).epsg(
            cfg.epsg_base
        )

        sql: CrownetSql = CrownetSql.from_dpmm_cfg(cfg)
        builder.set_map_type(cfg.map_type)

        return cfg.base_dir, builder, sql

    @classmethod
    def builder_from_output_folder(
        cls,
        data_root: str,
        hdf_file="data.h5",
        vec_name="vars_rep_0.vec",
        sca_name="vars_rep_0.sca",
        network_name="World",
        global_name="global.csv",
        map_glob="dcdMap_*.csv",
        epsg_base=Project.UTM_32N,
        override_hdf: bool = False,
    ) -> Tuple[str, DpmmHdfBuilder, CrownetSql]:
        logger.warning(
            "build_from_output_folder is deprecated. Use build_from_cfg instead. Try to guess configuration ..."
        )
        cfg = DpmmCfgCsv(
            base_dir=data_root,
            hdf_file=hdf_file,
            vec_name=vec_name,
            sca_name=sca_name,
            network_name=network_name,
            global_map_csv_name=global_name,
            node_map_csv_glob=map_glob,
            epsg_base=epsg_base,
        )
        try:
            _sql: CrownetSql = CrownetSql(
                vec_path=cfg.vec_path(),
                sca_path=cfg.sca_path(),
                network=cfg.network_name,
            )

            map_type, glb_map_path = _sql.guess_map_type()
        except Exception as e:
            logger.warning(
                f"assume map type {MapType.DENSITY} and ini path globalDensityMap"
            )
            map_type, glb_map_path = (MapType.DENSITY, "globalDensityMap")

        cfg.map_type = map_type
        cfg.global_map_ini_path = f"{cfg.network_name}.{glb_map_path}"

        return cls.builder_from_cfg(cfg, override_hdf=override_hdf)

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
