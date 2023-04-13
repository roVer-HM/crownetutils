from __future__ import annotations

import collections
import json
import shutil
import threading
from os.path import basename, join, split
from typing import Any

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import numpy as np
import pandas as pd
from dash import Dash, dash_table, dcc, html
from dash_extensions.javascript import Namespace, arrow_function
from flask_caching import Cache

import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.analysis.density_map import DensityMap
from roveranalyzer.analysis.hdfprovider.IHdfProvider import BaseHdfProvider
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.simulators.crownet.dcd.dcd_builder import DcdHdfBuilder
from roveranalyzer.simulators.vadere.plots.scenario import Scenario
from roveranalyzer.utils.misc import Project

dash_app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app.config.suppress_callback_exceptions = True
i_ = pd.IndexSlice

_decimal_04 = dash_table.Format.Format(precision=4)
_decimal_06 = dash_table.Format.Format(precision=4)
_col_types = {
    "measured_t": {"format": _decimal_06, "type": "numeric"},
    "received_t": {"format": _decimal_04, "type": "numeric"},
    "sourceHost": {"format": _decimal_04, "type": "numeric"},
    "sourceEntry": {"format": _decimal_06, "type": "numeric"},
    "owner_dist": {"format": _decimal_04, "type": "numeric"},
    "delay": {"format": _decimal_04, "type": "numeric"},
    "measurement_age": {"format": _decimal_04, "type": "numeric"},
    "update_age": {"format": _decimal_04, "type": "numeric"},
    "ymfD_d": {"format": _decimal_04, "type": "numeric"},
    "ymfD_t": {"format": _decimal_04, "type": "numeric"},
    "ymfD_t_old": {"format": _decimal_04, "type": "numeric"},
    "ymfD": {"format": _decimal_04, "type": "numeric"},
}


class Ctx:
    def __init__(self):
        self._data = {}

    def __getitem__(self, name: str) -> Any:
        if name not in self._data:
            self._data.setdefault(name, {})
        return self._data[name]

    def __setitem__(self, __name: str, __value: Any) -> None:
        self._data[__name] = __value

    def set(self, session, key: str, value: Any):
        _s = self.__getitem__(session)
        _s[key] = value
        self._data[session] = _s
        return _s

    def get(self, session, key: str, _default: Any):
        _s = self.__getitem__(session)
        if key not in _s:
            _s = self.set(session, key, _default)
        return _s[key]


class OppModel:
    def __init__(
        self, data_root: str, builder: DcdHdfBuilder, sql: OMNeT.CrownetSql
    ) -> None:
        self.data_root: str = data_root
        self.builder: DcdHdfBuilder = builder
        self.sql: OMNeT.CrownetSql = sql
        self.ctx: Ctx = Ctx()
        self.load()
        self._lock = threading.Lock()

    def reload(self, data_root: str):
        self.data_root = data_root
        r, b, s = OppAnalysis.builder_from_output_folder(self.data_root)
        self.builder = b
        self.sql = s
        self.load()

    @classmethod
    def asset_pdf_path(cls, root_path, relative=False, suffix=""):
        b_name = basename(root_path)
        if relative:
            p = join("assets", f"{b_name}{suffix}.pdf")
        else:
            p = join(dash_app.config.assets_folder, f"{b_name}{suffix}.pdf")
        return p

    @classmethod
    def copy_common_pdf(cls, data: dict, append_label=False):
        for path in data:
            pdf = join(path["value"], "common_output.pdf")
            if append_label:
                shutil.copyfile(
                    src=pdf, dst=cls.asset_pdf_path(path["value"], suffix=path["label"])
                )
            else:
                shutil.copyfile(src=pdf, dst=cls.asset_pdf_path(path["value"]))

    def load(self):
        self.pos: BaseHdfProvider = BaseHdfProvider(
            join(self.data_root, "trajectories.h5"), group="trajectories"
        )

        with self.builder.count_p.query as ctx:
            # all time points present
            self.count_index = ctx.select(key=self.builder.count_p.group, columns=[])
        self.cells = (
            self.count_index.reset_index(["simtime", "ID"], drop=True)
            .sort_index()
            .index.unique()
        )
        self.map_node_index = (
            self.count_index.index.get_level_values("ID").unique().sort_values()
        )
        self.map_time_index = (
            self.count_index.index.get_level_values("simtime").unique().sort_values()
        )

        self.host_ids = self.sql.host_ids()
        self.host_ids[0] = "Global View (Ground Truth)"
        self.host_ids = collections.OrderedDict(sorted(self.host_ids.items()))

        with self.builder.count_p.query as ctx:
            self.erroneous_cells = ctx.select(
                key=self.builder.count_p.group,
                where="(ID>0) & (err > 0)",
                columns=["count", "err"],
            )

        _mask = self.erroneous_cells["count"] == self.erroneous_cells["err"]
        self.erroneous_cells = self.erroneous_cells[_mask]
        _err_ret = (
            self.erroneous_cells.groupby(by=["x", "y"])
            .sum()
            .sort_values("count", ascending=False)
            .reset_index()
            .set_index(["x", "y"])
            .iloc[0:30]
            .copy()
        )
        _err_ret["node_ids"] = ""
        for g, _ in _err_ret.groupby(["x", "y"]):
            l = (
                self.erroneous_cells.groupby(by=["x", "y", "ID"])
                .sum()
                .loc[g]
                .nlargest(5, "err")
                .index.to_list()
            )
            _d = ", ".join([str(i) for i in l])
            _err_ret.loc[g, "node_ids"] = _d

        self.erroneous_cells = _err_ret.reset_index()

        b = pd.read_csv(join(self.data_root, "beacons.csv"), delimiter=";")
        if any(b["cell_x"] > 0x0FFF_FFFF) or any(b["cell_y"] > 0x0FFF_FFFF):
            raise ValueError("cell positions exceed 28 bit.")
        bound = self.builder.count_p.get_attribute("cell_bound")
        b["posY"] = bound[1] - b["posY"]  # fix translation !
        b["cell_id"] = (b["cell_x"].values << 28) + b["cell_y"]
        self.beacon_df = b

    def data_changed(self, data):
        # todo: make session aware!!!!
        try:
            self._lock.acquire()
            if self.data_root != data:
                print(f"load new data: {data}")
                self.reload(data)
                return True
            return False
        finally:
            self._lock.release()

    def topo_json(self, to_crs=Project.WSG84_lat_lon):
        name = split(self.sql.vadere_scenario)[-1]
        s = Scenario(join(self.data_root, f"vadere.d/{name}"))
        df = s.topography_frame(to_crs=to_crs)
        return json.loads(df.reset_index().to_json())

    def set_selected_node(self, session, value):
        self.ctx.set(session, "selected_node", value)

    def get_selected_node(self, session):
        return self.ctx.get(session, "selected_node", self.map_node_index[0])

    def get_node_tile_geojson_for(self, time_value, node_value):
        with self.pos.query as ctx:
            nodes = ctx.select(
                "trajectories",
                where=f"(time <= {time_value}) & (time > {time_value - 0.4})",
            )
        nodes = self.sql.apply_geo_position(
            nodes, Project.UTM_32N, Project.WSG84_lat_lon
        )
        nodes["tooltip"] = nodes["hostId"].astype(str) + " " + nodes["host"]
        nodes["color"] = "#0000bb"
        nodes["color"] = nodes["color"].where(
            nodes["hostId"] != node_value, other="#bb0000"
        )
        j = json.loads(nodes.reset_index().to_json())
        return j

    def get_cell_tile_geojson_for(self, time_value, node_id):
        # todo: cache map_data (Flask)
        map_data = self.builder.count_p.geo(Project.WSG84_lat_lon)[
            pd.IndexSlice[time_value, :, :, node_id]
        ]
        j = json.loads(map_data.reset_index().to_json())
        # j = json.loads(map_data.loc[time_value].reset_index().to_json())
        print(f"geojson for time {time_value} node {self.host_ids[node_id]}")
        return j

    def get_beacon_entry_exit(self, node_id: int, cell: tuple) -> pd.DataFrame:
        b = self.beacon_df

        c_size = self.builder.count_p.get_attribute("cell_size")
        beacon_mask = (
            (b["table_owner"] == node_id)
            & (b["posX"] > (cell[0] - c_size))
            & (b["posX"] < (cell[0] + 2 * c_size))
            & (b["posY"] > (cell[1] - c_size))
            & (b["posY"] < (cell[1] + 2 * c_size))
        )
        bs = b[beacon_mask]

        bs = bs.set_index(
            ["table_owner", "source_node", "event_time", "cell_x", "cell_y"]
        ).sort_index()
        bs["cell_change"] = 0
        bs["cell_change_count"] = 0
        bs["cell_change_cumsum"] = 0
        if bs.empty:
            return bs
        bs_missing = []
        bs_missing_idx = []
        for g, df in bs.groupby(by=["table_owner", "source_node"]):
            data = np.abs(df.index.to_frame()["cell_y"].diff()) + np.abs(
                df.index.to_frame()["cell_x"].diff()
            )
            data = data.fillna(1)
            data[data > 0] = -1
            bs.loc[df.index, ["cell_change"]] = data.astype(int)
            # use bs not data for change_log to use correct index..
            cell_change_at = data[data == -1].index
            change_iloc = []
            for _loc in cell_change_at:
                local_iloc = data.index.get_loc(_loc)
                global_iloc = bs.index.get_loc(_loc)
                if local_iloc > 0:
                    global_prev_iloc = bs.index.get_loc(data.index[local_iloc - 1])
                    change_iloc.append([global_iloc, global_prev_iloc])
                else:
                    change_iloc.append([global_iloc, None])

            cell_c_iloc = bs.columns.get_loc("cell_change")
            count_iloc = bs.columns.get_loc("cell_change_count")
            cell_id_iloc = bs.columns.get_loc("cell_id")
            for g_iloc, g_prev_iloc in change_iloc:
                if g_prev_iloc is not None:
                    # copy g_iloc,
                    # replace cell_x, cell_y, cell_id with data from g_prev_iloc
                    # add cell_id of g_iloc as positive value in 'cell_change'
                    # do not change g_prev_iloc
                    idx = bs.index[g_iloc]
                    idx_prev = bs.index[g_prev_iloc]
                    missing = bs.iloc[g_iloc].copy()
                    missing_idx = (idx[0], idx[1], idx[2], idx_prev[3], idx_prev[4])
                    missing["event"] = "move_out"
                    missing["cell_id"] = bs.iloc[g_prev_iloc, cell_id_iloc]
                    missing["cell_change"] = bs.iloc[g_iloc, cell_id_iloc]
                    missing["cell_change_count"] = -1
                    bs_missing.append(missing.to_list())
                    bs_missing_idx.append(missing_idx)

                    # negative cell_id: 'node comes from this cell i.e. it was there in the previous step' -> thus increment count now!
                    #
                    bs.iloc[g_iloc, cell_c_iloc] = (
                        -1 * bs.iloc[g_prev_iloc, cell_id_iloc]
                    )
                    bs.iloc[g_iloc, count_iloc] = 1
                else:
                    # first occurrence no idea where that nodes was before set -1
                    bs.iloc[g_iloc, cell_c_iloc] = -1
                    bs.iloc[g_iloc, count_iloc] = 1
        missing_df = pd.DataFrame(
            bs_missing,
            index=pd.MultiIndex.from_tuples(bs_missing_idx, names=bs.index.names),
            columns=bs.columns,
        )
        bs = pd.concat([bs, missing_df])
        # remove surrounding cells (copy!)
        bs = bs.loc[pd.IndexSlice[:, :, :, cell[0], cell[1]], :].copy()

        bs = (
            bs.reset_index()
            .set_index(["table_owner", "event_time", "source_node"])
            .sort_index()
        )
        for g, df in bs.groupby(by=["table_owner", "cell_x", "cell_y"]):
            bs.loc[df.index, ["cell_change_cumsum"]] = df["cell_change_count"].cumsum()
        # _m = (bs["cell_x"] == int(cell[0])) & (bs["cell_y"] == int(cell[1]))
        bs = bs.reset_index("source_node")
        return bs

    def get_measurements(self, time, node_id, cell_id, alpha=0.5):

        x, y = self.cells[cell_id]
        df = self.builder.map_p[
            pd.IndexSlice[float(time), float(x), float(y), :, node_id], :
        ]
        dist_sum = df["sourceEntry"].sum()
        time_sum = df["measurement_age"].sum()
        # df["ymfD_t_old"] = alpha*(df["measurement_age"]/time_sum)

        t_min = df["measurement_age"].min()
        time_sum_new = (
            df["measurement_age"].sum() - df["measurement_age"].shape[0] * t_min
        )

        df["ymfD_t"] = alpha * ((df["measurement_age"] - t_min) / time_sum_new)
        df["ymfD_d"] = (1 - alpha) * (df["sourceEntry"] / dist_sum)
        df["ymfD"] = df["ymfD_t"] + df["ymfD_d"]

        return df

    def cell_ymfD_value(self, df: pd.DataFrame, alpha=0.5):

        sum_df: pd.DataFrame = (
            df[["measurement_age", "sourceEntry"]]
            .groupby(by=["simtime"])
            .agg(["sum", "min", "max", "count"])
        )
        sum_df[("measurement_age", "sum_min_norm")] = (
            sum_df[("measurement_age", "sum")]
            - sum_df[("measurement_age", "min")] * sum_df[("measurement_age", "count")]
        )
        sum_df = sum_df.sort_index(axis=1)
        sum_df.columns = [f"{l0}_{l1}" for l0, l1 in sum_df.columns]
        sum_df = sum_df.drop(columns=["sourceEntry_count"])
        sum_df = sum_df.rename(columns={"measurement_age_count": "count"})
        _df = sum_df

        df["ymfD_t"] = 0.0
        df["ymfD_d"] = 0.0
        df["ymfD"] = 0.0
        for time, _df in df.groupby(by=["simtime"]):
            df.loc[_df.index, ["ymfD_t"]] = (
                alpha
                * (_df["measurement_age"] - df.loc[time, ["measurement_age_min"]][0])
                / df.loc[time, ["measurement_age_sum_min_norm"]][0]
            )
            df.loc[_df.index, ["ymfD_t"]] = (
                (1 - alpha)
                * (_df["sourceEntry"])
                / df.loc[time, ["sourceEntry_sum"]][0]
            )

        df["ymfD"] = df["ymfD_t"] + df["ymfD_d"]

        return sum_df, df


class _DashUtil:
    @classmethod
    def get_colorbar(
        cls, width=300, height=30, position="bottomleft", value="count", **kwargs
    ):
        if value == "count":
            classes = [0, 1, 2, 3, 4, 5, 6, 7]
            colorscale = [
                "#E5E5E5",
                "#FED976",
                "#FEB24C",
                "#FD8D3C",
                "#FC4E2A",
                "#E31A1C",
                "#BD0026",
                "#800026",
            ]
            ctg = [
                "{}".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])
            ] + ["{}+".format(classes[-1])]
        elif value == "err":
            classes = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
            colorscale = [
                "#083877",
                "#0055FF",
                "#3080BD",
                "#57A0CE",
                "#8DC1DD",
                "#B5D4E9",
                "#E5EFF9",
                "#E5E5E5",  # 0
                "#FED976",
                "#FEB24C",
                "#FD8D3C",
                "#FC4E2A",
                "#E31A1C",
                "#BD0026",
                "#800026",
            ]
            ctg = [
                "{}<".format(classes[0]),
                *[
                    "{}".format(cls, classes[i + 1])
                    for i, cls in enumerate(classes[1:-1])
                ],
                "{}>".format(classes[-1]),
            ]
        else:
            classes = [0, 1, 4, 9, 16, 32]
            colorscale = [
                "#E5E5E5",  # 0
                "#FED976",
                "#FEB24C",
                "#FD8D3C",
                "#FC4E2A",
                "#E31A1C",
            ]
            ctg = [
                "{}".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])
            ] + ["{}+".format(classes[-1])]

        style = dict(weight=2, opacity=1, fillColor="red", fillOpacity=0.6)
        colorbar = dlx.categorical_colorbar(
            categories=ctg,
            colorscale=colorscale,
            width=width,
            height=height,
            position=position,
            **kwargs,
        )
        return classes, colorscale, colorbar, style

    _colorbar_style_clb = """function(feature, context){
            const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
            const value = feature.properties[colorProp];  // get value the determines the color
            var ret = classes.findIndex(e => e == value);
            if (ret == -1){
                if (value < classes[0]){
                    ret = 0;
                } else {
                    ret = classes[classes.length-1];
                }
            }
            style.fillColor = colorscale[ret];
            style.color = colorscale[ret];
            return style;
        }
    """

    _poly_style_clb = """function(feature, context){
            return feature.properties['style'];
        }
    """

    _node_point_to_layer = """function(feature, latlng, context){
        const {circleOptions, colorProp} = context.props.hideout;
        //const csc = chroma.scale(colorscale).domain([min, max]);  // chroma lib to construct colorscale
        circleOptions.fillColor =  feature.properties[colorProp] //csc(feature.properties[colorProp]);  // set color based on color prop.
        return L.circleMarker(latlng, circleOptions);  // sender a simple circle marker.
        }
    """

    @classmethod
    def cols(cls, cols, ignore=()):
        return [
            dict(id=i, name=i, **_col_types.get(i, {})) for i in cols if i not in ignore
        ]

    @classmethod
    def build_measurement_tbl(cls, m: OppModel, tbl_id, slider_id, h3_id):
        return [
            dbc.Row(dbc.Col(html.H4(id=h3_id))),
            dbc.Row(
                dbc.Col(
                    dash_table.DataTable(
                        id=tbl_id,
                        page_action="none",
                        filter_action="native",
                        sort_action="native",
                        sort_mode="multi",
                        fixed_rows={"headers": True},
                        style_table=dict(
                            height="600px", overflowY="auto", overflowX="scroll"
                        ),
                    )
                )
            ),
            dbc.Row(dbc.Col(cls.build_time_slider(m, slider_id))),
        ]

    @classmethod
    def build_time_slider(cls, m: OppModel, slider_id):
        opt = {
            k: {"label": f"{k}s", "style": {"display": "none"}}
            for k in m.map_time_index
        }
        for k in range(0, m.map_time_index.shape[0], 10):
            t = m.map_time_index[k]
            del opt[t]["style"]["display"]
        # del m[self.m.map_time_index[-1]]["style"]["display"]
        return dcc.Slider(
            # 0, 100, 10,
            step=None,
            id=slider_id,
            marks=opt,
            value=m.map_time_index[0],
            # dots= False,
            included=False,
            tooltip={"placement": "bottom", "always_visible": True},
        )

    @classmethod
    def module_header(cls, id, **kwargs):
        return dbc.Row(dbc.Col([html.H2(id=id, **kwargs)]))

    @classmethod
    def map_view(cls, ns: Namespace, id_builder):
        topo_overlays = cls.build_topography_layer(ns, id_builder)
        cell_overlays = cls.build_cell_layer(ns, id_builder)
        node_overlays = cls.build_node_layer(ns, id_builder)

        return dl.Map(
            dl.LayersControl(
                [
                    *DensityMap.get_dash_tilelayer(),
                    *cell_overlays,
                    *node_overlays,
                    *topo_overlays,
                ]
            ),
            center=(48.162, 11.586),
            zoom=18,
            style={
                "width": "100%",
                "height": "65vh",
                "background-color": "white",
                "margin": "auto",
                "display": "block",
            },
        )

    @classmethod
    def build_node_layer(cls, ns: Namespace, id_builder):

        clb = ns(ns.add(cls._node_point_to_layer, name="node_point_to_layer"))

        nodes = dl.GeoJSON(
            id=id_builder("node_tiles"),
            format="geojson",
            zoomToBounds=False,
            options=dict(pointToLayer=clb),
            hideout=dict(
                colorProp="color",
                circleOptions=dict(fillOpacity=1.0, stroke=False, radius=3),
            ),
        )
        overlays = []

        overlays.append(
            dl.Overlay(
                dl.LayerGroup(nodes),
                checked=True,
                name="nodes",
            )
        )
        return overlays

    @classmethod
    def build_topography_layer(cls, ns: Namespace, id_builder):
        topo_clb = ns(ns.add(cls._poly_style_clb, name="ploy_styler"))
        topo = dl.GeoJSON(
            id=id_builder("topo_tiles"),
            options=dict(style=topo_clb),
            zoomToBounds=False,
            format="geojson",
            # hoverStyle=arrow_function(dict(weight=2, color="#222", dashArray="")),
            hideout=dict(style=dict(), styleProp=["fillColor"]),
        )
        return [
            dl.Overlay(
                dl.LayerGroup(topo),
                checked=True,
                name="topo",
            )
        ]

    @classmethod
    def build_cell_layer(cls, ns: Namespace, id_builder):

        # colorbar
        classes, colorscale, colorbar, style = DashUtil.get_colorbar(
            id=id_builder("map-colorbar")
        )
        # cell color style
        style_clb = ns(ns.add(cls._colorbar_style_clb, name="map_style_clb"))
        cells = dl.GeoJSON(
            id=id_builder("cell_tiles"),
            options=dict(style=style_clb),
            zoomToBounds=False,
            zoomToBoundsOnClick=True,
            format="geojson",
            hoverStyle=arrow_function(dict(weight=2, color="#222", dashArray="")),
            hideout=dict(
                colorscale=colorscale, classes=classes, style=style, colorProp="count"
            ),
        )

        # cell over info box
        info = html.Div(
            id=id_builder("cell_info"),
            className="info-box",
            style={
                "position": "absolute",
                "bottom": "100px",
                "left": "10px",
                "z-index": "1000",
            },
        )

        overlays = []
        overlays.append(
            dl.Overlay(
                dl.LayerGroup(cells),
                checked=True,
                name="cells",
            )
        )
        overlays.append(
            dl.Overlay(
                dl.LayerGroup(colorbar, id=id_builder("map-colorbar-wrapper")),
                checked=True,
                name="colorbar",
            )
        )
        overlays.append(dl.Overlay(dl.LayerGroup(info), checked=True, name="info"))

        return overlays

    def get_map_info(self, feature=None):
        header = [html.H4("Cell Info")]
        if not feature:
            return header + [html.P("Hoover over a cell")]
        return header + [
            html.B(
                f"[{feature['properties']['cell_x']}, {feature['properties']['cell_y']}]"
            ),
            html.Br(),
            f"Count: {feature['properties']['count']}",
        ]

    @classmethod
    def get_cell_info(cls, feature, storage, **kwargs):
        storage = {} if storage is None else storage
        header = [html.H4("Cell Info")]
        if feature is None and "last_cell_hovered" not in storage:
            return header + [html.P("Hoover over a cell")], storage
        else:
            if feature is None:
                feature = storage["last_cell_hovered"]
            else:
                storage["last_cell_hovered"] = feature
            ret = header + [
                html.B(
                    f"[{feature['properties']['cell_x']}, {feature['properties']['cell_y']}]"
                ),
                html.Br(),
                f"Count: {feature['properties']['count']}",
                html.Br(),
                f"Error: {feature['properties']['err']}",
                html.Br(),
                f"Owner Distance: {feature['properties']['owner_dist']:000.3f} m",
            ]
            return ret, storage

    @classmethod
    def cell_dropdown(cls, id, m: OppModel, **kwargs):
        cell_map = {k: tuple(v) for k, v in enumerate(m.cells)}

        return dcc.Dropdown(
            id=id,
            options=[
                {"value": k, "label": f"[{v[0]},{v[1]}]"} for k, v in cell_map.items()
            ],
            multi=False,
            searchable=True,
            value=0,
            **kwargs,
        )

    @classmethod
    def id_dropdown(cls, id, m: OppModel, **kwargs):

        return dcc.Dropdown(
            id=id,
            options=[
                {"value": k, "label": f"{k} - {v}"} for k, v in m.host_ids.items()
            ],
            multi=False,
            searchable=True,
            value=0,
            style={"position": "relative", "zIndex": "999"},
            **kwargs,
        )

    @classmethod
    def data_dropdown(cls, id, m: OppModel, opt):

        return dcc.Dropdown(
            id=id,
            options=opt,
            multi=False,
            searchable=True,
            value=opt[-1]["value"],
            style={"position": "relative", "zIndex": "999"},
        )


DashUtil = _DashUtil()
