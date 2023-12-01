import unittest
from unittest.mock import MagicMock, call, patch

import pandas as pd

from crownetutils.analysis.dpmm.dpmm_cfg import DpmmCfg, DpmmCfgCsv
from crownetutils.omnetpp.scave import CrownetSql


class CrownetSqlTest(unittest.TestCase):
    def test_default(self):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")

        self.assertEqual(sql.module_vectors, ["misc", "pNode", "vNode"])
        self.assertEqual(sql.network, "World")

    def test_default_with_cfg(self):
        cfg = DpmmCfgCsv(
            base_dir="/temp/foo",
            network_name="Net",
            map_app_path="map",
            beacon_app_path="beacon",
            module_vectors=["foo"],
        )
        sql = CrownetSql.from_dpmm_cfg(cfg=cfg)

        self.assertEqual(sql.module_vectors, ["foo"])
        self.assertEqual(sql.network, "Net")

    def test_module_path_redirect_with_cfg(self):
        cfg = DpmmCfgCsv(
            base_dir="/temp/foo",
            network_name="Net",
            map_app_path="map",
            beacon_app_path="beacon",
            module_vectors=["foo"],
        )
        sql = CrownetSql.from_dpmm_cfg(cfg=cfg)

        b1 = sql.m_beacon().info_str()
        self.assertEqual(b1, "or[Net.foo[%].beacon.app]")
        b1 = sql.m_beacon(path="scheduler").info_str()
        self.assertEqual(b1, "or[Net.foo[%].beacon.scheduler]")
        b1 = sql.m_beacon(path=".scheduler").info_str()
        self.assertEqual(b1, "or[Net.foo[%].beacon.scheduler]")

        m1 = sql.m_map().info_str()
        self.assertEqual(m1, "or[Net.foo[%].map.app]")
        m1 = sql.m_map(path="scheduler").info_str()
        self.assertEqual(m1, "or[Net.foo[%].map.scheduler]")
        m1 = sql.m_map(path=".scheduler").info_str()
        self.assertEqual(m1, "or[Net.foo[%].map.scheduler]")

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_beacon_at_app_0(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeBeaconApp"], columns=["paramValue"])
        ]
        b1 = sql.m_beacon().info_str()
        self.assertEqual(
            b1,
            "or[World.misc[%].app[0].app, World.pNode[%].app[0].app, World.vNode[%].app[0].app]",
        )
        parameter_data_beacon.assert_called_once()

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_beacon_at_app_1(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeAppNotContainingBea_con"], columns=["paramValue"]),
            pd.DataFrame(["SomeBeaconApp"], columns=["paramValue"]),
        ]
        b1 = sql.m_beacon(path="X").info_str()
        self.assertEqual(
            b1,
            "or[World.misc[%].app[1].X, World.pNode[%].app[1].X, World.vNode[%].app[1].X]",
        )

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_beacon_with_no_beacon_in_typename(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeAppNotContainingBea_con"], columns=["paramValue"]),
            pd.DataFrame(["SomeAppNotContainingBea_con"], columns=["paramValue"]),
        ]
        try:
            b1 = sql.m_beacon(path="X").info_str()
            self.fail("no beacon app found. should fail.")
        except Exception:
            pass

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_map_at_app_0(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeDensityMapApp"], columns=["paramValue"])
        ]
        b1 = sql.m_map().info_str()
        self.assertEqual(
            b1,
            "or[World.misc[%].app[0].app, World.pNode[%].app[0].app, World.vNode[%].app[0].app]",
        )
        parameter_data_beacon.assert_called_once()

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_map_at_app_1(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeOhterApp"], columns=["paramValue"]),
            pd.DataFrame(["SomeDensityMapApp"], columns=["paramValue"]),
        ]
        b1 = sql.m_map(path="Y.Z").info_str()
        self.assertEqual(
            b1,
            "or[World.misc[%].app[1].Y.Z, World.pNode[%].app[1].Y.Z, World.vNode[%].app[1].Y.Z]",
        )

    @patch("crownetutils.omnetpp.scave.CrownetSql.parameter_data")
    def test_module_path_redirect_without_cfg_beacon_with_no_map_in_typename(
        self, parameter_data_beacon: MagicMock
    ):
        sql = CrownetSql("/tmp/foo.vec", "/tmp/foo.sca")
        parameter_data_beacon.side_effect = [
            pd.DataFrame(["SomeAppNotContainingM_ap"], columns=["paramValue"]),
            pd.DataFrame(["SomeAppNotContainingM_ap"], columns=["paramValue"]),
        ]
        try:
            b1 = sql.m_map().info_str()
            self.fail("no map app found. should fail.")
        except Exception:
            pass
