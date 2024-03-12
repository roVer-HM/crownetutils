import unittest

from crownetutils.analysis.dpmm import MapType
from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfgCsv
from crownetutils.utils.misc import Project


class DpmmCfgTest(unittest.TestCase):
    def test_default_setup(self):
        cfg = DpmmCfgCsv(base_dir="/tmp/foo")
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
        self.assertEqual(cfg.beacon_app_path, "app[0]")
        self.assertEqual(cfg.map_app_path, "app[1]")

        self.assertEqual(cfg.hdf_path(), "/tmp/foo/data.h5")
        self.assertEqual(cfg.vec_path(), "/tmp/foo/vars_rep_0.vec")
        self.assertEqual(cfg.sca_path(), "/tmp/foo/vars_rep_0.sca")

    def test_single_module(self):
        cfg = DpmmCfgCsv(base_dir="/tmp/foo", module_vectors=["bar"])

        b1 = cfg.m_beacon().info_str()
        self.assertEqual(b1, "or[World.bar[%].app[0].app]")
        b2 = cfg.m_beacon(modules=["foo"]).info_str()
        self.assertEqual(b2, "or[World.foo[%].app[0].app]")
        b3 = cfg.m_beacon(node_index=13).info_str()
        self.assertEqual(b3, "or[World.bar[13].app[0].app]")

        b1 = cfg.m_map().info_str()
        self.assertEqual(b1, "or[World.bar[%].app[1].app]")
        b2 = cfg.m_map(modules=["foo"]).info_str()
        self.assertEqual(b2, "or[World.foo[%].app[1].app]")
        b3 = cfg.m_map(node_index=13).info_str()
        self.assertEqual(b3, "or[World.bar[13].app[1].app]")

    def test_paths(self):
        cfg = DpmmCfgCsv(
            base_dir="/tmp/foo",
            module_vectors=["bar"],
            map_app_path="M",
            beacon_app_path="B",
        )

        b1 = cfg.m_beacon(modules=["foo"], path="baz").info_str()
        self.assertEqual(b1, "or[World.foo[%].B.baz]")

        b2 = cfg.m_beacon(modules=["foo"], node_index=5, path="baz").info_str()
        self.assertEqual(b2, "or[World.foo[5].B.baz]")

    def test_paths(self):
        cfg = DpmmCfgCsv(
            base_dir="/tmp/foo",
            module_vectors=["bar"],
            map_app_path="M",
            beacon_app_path=None,
        )
        try:
            cfg.m_beacon()
            self.fail("no beacon defined. Should fail")
        except Exception as e:
            pass

    def test_multi_module(self):
        cfg = DpmmCfgCsv(
            network_name="A",
            base_dir="/tmp/foo",
            module_vectors=["foo", "bar"],
            beacon_app_path="x[0].y",
        )

        b1 = cfg.m_beacon()
        self.assertEqual(b1.info_str(), "or[A.foo[%].x[0].y.app, A.bar[%].x[0].y.app]")

    def test_multi_module_dict_app(self):
        cfg = DpmmCfgCsv(
            network_name="A",
            base_dir="/tmp/foo",
            module_vectors=["foo", "bar"],
            map_app_path={"foo": "x[0].y", "bar": "w[1].v"},
            beacon_app_path="app[99]",
        )

        b1 = cfg.m_beacon()
        self.assertEqual(
            b1.info_str(), "or[A.foo[%].app[99].app, A.bar[%].app[99].app]"
        )

        m1 = cfg.m_map(path="XXX")
        self.assertEqual(m1.info_str(), "or[A.foo[%].x[0].y.XXX, A.bar[%].w[1].v.XXX]")

        m2 = cfg.m_map(node_index="5..10")
        self.assertEqual(
            m2.info_str(), "or[A.foo[5..10].x[0].y.app, A.bar[5..10].w[1].v.app]"
        )

        try:
            m3 = cfg.m_map(modules=["foo", "bazz"])
            self.fail("bazz should not exist")
        except KeyError:
            pass


if __name__ == "__main__":
    unittest.main()
