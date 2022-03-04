from os.path import join
from typing import Tuple

import roveranalyzer.simulators.crownet.dcd as Dcd
import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.utils import Project


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
    ) -> Tuple[str, Dcd.DcdHdfBuilder, OMNeT.CrownetSql]:

        builder = Dcd.DcdHdfBuilder.get(hdf_file, data_root).epsg(epsg_base)

        sql = OMNeT.CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
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
    ) -> Tuple[str, Dcd.DcdHdfBuilder, OMNeT.CrownetSql]:

        data_root = join(
            data_root, "simulation_runs/outputs", f"Sample_{parameter_id}_{run_id}", out
        )
        builder = Dcd.DcdHdfBuilder.get(hdf_file, data_root).epsg(epsg_base)

        sql = OMNeT.CrownetSql(
            vec_path=f"{data_root}/{vec_name}",
            sca_path=f"{data_root}/{sca_name}",
            network=network_name,
        )
        return data_root, builder, sql

    @staticmethod
    def find_selection_method(builder: Dcd.DcdHdfBuilder):
        p = builder.build().map_p
        return p.get_attribute("used_selection")
