import unittest

from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg, MapType
from crownetutils.utils.misc import Project


class DpmmCfgTest(unittest.TestCase):
    def test_default_setup(self):
        cfg = DpmmCfg(base_dir="/tmp/foo")
        self.assertEqual(cfg.hdf_file, "data.h5")
        self.assertEqual(cfg.vec_name, "vars_rep_0.vec")
        self.assertEqual(cfg.sca_name, "vars_rep_0.sca")
        self.assertEqual(cfg.network_name, "World")
        self.assertEqual(cfg.map_type, MapType.DENSITY)
        self.assertEqual(cfg.global_map_ini_path, "World.globalDensityMap")
        self.assertEqual(cfg.global_map_csv_name, "global.csv")
        self.assertEqual(cfg.node_map_csv_glob, "dcdMap_*.csv")
        self.assertEqual(cfg.epsg_base, Project.UTM_32N)
        self.assertEqual(cfg.module_vectors, ["misc", "pNode", "vNode"])
        self.assertEqual(cfg.beacon_app_sql_op, "app[0].app")
        self.assertEqual(cfg.map_app_sql_op, "app[1].app")

        self.assertEqual(cfg.hdf_path(), "/tmp/foo/data.h5")
        self.assertEqual(cfg.vec_path(), "/tmp/foo/vars_rep_0.vec")
        self.assertEqual(cfg.sca_path(), "/tmp/foo/vars_rep_0.sca")

    def test_single_module(self):
        cfg = DpmmCfg(base_dir="/tmp/foo", module_vectors=["bar"])

        b1 = cfg.get_beacon_app_sql_op().info_str()
        self.assertEqual(b1, "or[World.bar[%].app[0].app]")
        b2 = cfg.get_beacon_app_sql_op(modules=["foo"]).info_str()
        self.assertEqual(b2, "or[World.foo[%].app[0].app]")
        b3 = cfg.get_beacon_app_sql_op(node_index=13).info_str()
        self.assertEqual(b3, "or[World.bar[13].app[0].app]")

        b1 = cfg.get_map_app_sql_op().info_str()
        self.assertEqual(b1, "or[World.bar[%].app[1].app]")
        b2 = cfg.get_map_app_sql_op(modules=["foo"]).info_str()
        self.assertEqual(b2, "or[World.foo[%].app[1].app]")
        b3 = cfg.get_map_app_sql_op(node_index=13).info_str()
        self.assertEqual(b3, "or[World.bar[13].app[1].app]")

    def test_multi_module(self):
        cfg = DpmmCfg(
            network_name="A",
            base_dir="/tmp/foo",
            module_vectors=["foo", "bar"],
            beacon_app_sql_op="x[0].y",
        )

        b1 = cfg.get_beacon_app_sql_op()
        self.assertEqual(b1.info_str(), "or[A.foo[%].x[0].y, A.bar[%].x[0].y]")

    def test_multi_module_dict_app(self):
        cfg = DpmmCfg(
            network_name="A",
            base_dir="/tmp/foo",
            module_vectors=["foo", "bar"],
            map_app_sql_op={"foo": "x[0].y", "bar": "w[1].v"},
        )

        b1 = cfg.get_map_app_sql_op()
        self.assertEqual(b1.info_str(), "or[A.foo[%].x[0].y, A.bar[%].w[1].v]")

        b1 = cfg.get_map_app_sql_op(node_index="5..10")
        self.assertEqual(b1.info_str(), "or[A.foo[5..10].x[0].y, A.bar[5..10].w[1].v]")

        try:
            b2 = cfg.get_map_app_sql_op(modules=["foo", "bazz"])
            self.fail("bazz should not exist")
        except KeyError:
            pass


if __name__ == "__main__":
    unittest.main()
