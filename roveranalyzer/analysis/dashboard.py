from __future__ import annotations

import json
from re import I

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from click import option
from dash import callback_context, dash_table, dcc, html
from dash.dependencies import Input, Output
from dash_extensions.javascript import Namespace, arrow_function
from mercantile import children

import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.analysis.dashapp import DashUtil, OppModel, dash_app
from roveranalyzer.analysis.density_map import DensityMap
from roveranalyzer.simulators.crownet.dcd.dcd_builder import DcdHdfBuilder
from roveranalyzer.utils.general import Project


class _DashApp:
    def __init__(
        self, name, id_prefix, m: OppModel, ns: Namespace | None = None
    ) -> None:
        self.name = name
        self.id_prefix = id_prefix
        # self.app = Dash(__name__)
        self.fig = None
        self.m = m
        self.ns = Namespace("dashExtensions", "default") if ns is None else ns

    def get_layout(self, with_container: bool = False):
        pass

    def init(self):
        pass

    def run_app(self):
        self.init()
        dash_app.layout = self.get_layout(True)
        self.ns.dump(assets_folder=dash_app.config.assets_folder)
        dash_app.run_server(debug=True, use_reloader=False)

    def id(self, name):
        return f"{self.id_prefix}_{name}"


class CellErrorInsepctor(_DashApp):
    def __init__(self, m: OppModel) -> None:
        super().__init__("Cell Error Inspector", "cell", m)

        dash_app.callback(
            Output(self.id("cell-node-id-dropdown"), "options"),
            [Input(self.id("cell-dropdown"), "value")],
        )(self.clb_update_id_dropdown)

        dash_app.callback(
            Output(self.id("cell-err-graph"), "figure"),
            [Input(self.id("cell-node-id-dropdown"), "value")],
        )(self.clb_update_error_figure)

    def get_layout(self, with_container: bool = False):

        layout = html.Div(
            children=[
                DashUtil.module_header(
                    id=self.id("cell_err_title"), children=["Cell Error View"]
                ),
                dbc.Row(
                    [
                        dbc.Col(width=3, children=[html.Div("Select Cell:")]),
                        dbc.Col(
                            DashUtil.cell_dropdown(self.id("cell-dropdown"), self.m)
                        ),
                    ],
                    align="center",
                    className="map-ctrl",
                ),
                dbc.Row(
                    [
                        dbc.Col(width=3, children=[html.Div("Select Id:")]),
                        dbc.Col(
                            DashUtil.id_dropdown(
                                self.id("cell-node-id-dropdown"), self.m
                            )
                        ),
                    ],
                    align="center",
                    className="map-ctrl",
                ),
                dbc.Row(dbc.Col(dcc.Graph(id=self.id("cell-err-graph")))),
            ],
            id=self.id("wrapper"),
            className="module-wrapper",
        )
        if with_container:
            layout = dbc.Container(layout)
        return layout

    def update_map_node(self, value):
        node_id = 0 if value is None else value
        if isinstance(value, list) and len(value) > 0:
            node_id = value[0]
        else:
            node_id = 0
        print(f"get map for id {node_id}")
        self.cells = self.b.count_p.geo(Project.WSG84_lat_lon)[
            pd.IndexSlice[int(self.time_index[0]), :, :, node_id]
        ]
        # self.nodes = self.s.host_position(
        #     epsg_code_base=Project.UTM_32N,
        #     # epsg_code_to=Project.OpenStreetMaps,
        #     epsg_code_to=Project.WSG84_lat_lon,
        #     time_slice=slice(324.8),
        # )

    # callback
    def clb_update_error_figure(self, value):
        _mask = self.ca["ID"] == 0
        if value is not None:
            value = [value] if isinstance(value, int) else value
            for v in value:

                _mask = _mask | (self.ca["ID"] == v)

        fig = px.scatter(
            self.ca[_mask],
            x="simtime",
            y="count",
            color="id",
            hover_data=["id", "count", "err"],
        )
        fig.update_xaxes(
            range=[self.m.map_time_index.min(), self.m.map_time_index.max()]
        )
        return fig

    # callback
    def clb_update_id_dropdown(self, value, *args, **kwargs):

        self.read_ids_for_cell_index(value)
        ids = self.ca["ID"].unique()
        ids = ids[ids > 0]
        return [{"value": int(k), "label": f"{k} - {self.m.host_ids[k]}"} for k in ids]

    def read_ids_for_cell_index(self, value, *args, **kwargs):
        i = pd.IndexSlice
        value = 0 if value is None else value
        _v = self.m.cells[value]
        with self.m.builder.count_p.query as ctx:
            ca = ctx.select(
                key=self.m.builder.count_p.group,
                where=f"(x == {float(_v[0])}) & (y == {float(_v[1])})",
            )
        self.ca = ca.reset_index(["x", "y"]).sort_index()

        ca_0 = self.ca.loc[i[:, 0], :]
        idx_diff = pd.Index(
            zip(self.m.map_time_index, np.zeros(self.m.map_time_index.shape[0]))
        ).difference(ca_0.index)
        _add_zero = pd.DataFrame(index=idx_diff, columns=ca_0.columns).fillna(0)
        self.ca = pd.concat([self.ca, _add_zero])
        self.ca = self.ca.reset_index()
        self.ca["id"] = self.ca["ID"].astype("str")

    def init(self):
        self.cwerr = self.m.erroneous_cells
        self.read_ids_for_cell_index(value=None)  # todo...


class _MapView(_DashApp):
    def __init__(self, m: OppModel, ns: Namespace | None = None) -> None:
        super().__init__("Map View", "map", m)

        # ### callbacks
        dash_app.callback(
            Output(self.id("geojson"), "data"), Input(self.id("time_slider"), "value")
        )(self.clb_map_time_slider)
        dash_app.callback(
            Output(self.id("map-time-in"), "value"),
            Input(self.id("time_slider"), "value"),
        )(self.clb_map_time_input)
        dash_app.callback(
            Output(self.id("map_title"), "children"),
            Input(self.id("map_node_input"), "value"),
        )(self.clb_map_node_input)
        dash_app.callback(
            Output(self.id("cell_info"), "children"),
            Input(self.id("geojson"), "hover_feature"),
        )(DashUtil.get_cell_info)

    def clb_map_time_slider(self, value):
        return self.m.get_geojson_for(value)

    def clb_map_time_input(self, value):
        return value

    def clb_map_node_input(self, value):
        value = 0 if value is None else value
        changed = self.m.set_selected_node(value)
        print(f"node value changed: {changed}")
        return f"Density Map for Node {value} - {self.m.host_ids.get(value, '???')}"

    def clb_bar(self, value):
        # print(f"got drag value: {value}")
        return value

    def get_layout(self, with_container: bool = False):
        print("build layout")
        layout = html.Div(
            children=[
                DashUtil.module_header(self.id("map_title")),
                dbc.Row(
                    [
                        dbc.Col(width=3, children=[html.Div("Select Node:")]),
                        dbc.Col(
                            DashUtil.id_dropdown(self.id("map_node_input"), self.m)
                        ),
                    ],
                    align="center",
                    className="map-ctrl",
                ),
                dbc.Row(dbc.Col([self._build_map_layout(id_prefix=self.id_prefix)])),
                dbc.Row(
                    [
                        dbc.Col(
                            width=1,
                            children=[
                                dcc.Input(
                                    id=self.id("map-time-in"), style={"width": "100%"}
                                )
                            ],
                        ),
                        dbc.Col(self._build_map_time_slider(id_prefix=self.id_prefix)),
                    ],
                    className="map-ctrl",
                    align="center",
                ),
            ],
            id=self.id("wrapper"),
            className="module-wrapper",
            style={
                "width": "80%",
                "margin": "auto",
                "display": "block",
                "position": "relative",
            },
        )
        if with_container:
            layout = dbc.Container(layout)

        return layout

    def _build_map_time_slider(self, id_prefix="map"):
        m = {
            k: {"label": f"{k}s", "style": {"display": "none"}}
            for k in self.m.map_time_index
        }
        for k in range(0, self.m.map_time_index.shape[0], 10):
            t = self.m.map_time_index[k]
            del m[t]["style"]["display"]
        # del m[self.m.map_time_index[-1]]["style"]["display"]
        return dcc.Slider(
            # 0, 100, 10,
            step=None,
            id=f"{id_prefix}_time_slider",
            marks=m,
            value=self.m.map_time_index[0],
            # dots= False,
            included=False,
            tooltip={"placement": "bottom", "always_visible": True},
        )

    def _build_map_layout(
        self,
        id_prefix="map",
    ):
        # colorbar
        classes, colorscale, colorbar = DashUtil.get_colorbar(
            id=f"{id_prefix}_colorbar"
        )
        # cell color style
        style, style_clb = DashUtil.get_map_style(self.ns)
        geoj = dl.GeoJSON(
            options=dict(style=self.ns(style_clb)),
            zoomToBounds=False,
            zoomToBoundsOnClick=True,
            format="geojson",
            hoverStyle=arrow_function(dict(weight=3, color="#222", dashArray="")),
            hideout=dict(
                colorscale=colorscale, classes=classes, style=style, colorProp="count"
            ),
            id=f"{id_prefix}_geojson",
        )
        # cell over info box
        info = html.Div(
            id=f"{id_prefix}_cell_info",
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
                dl.LayerGroup(geoj),
                checked=True,
                name="cells",
            )
        )
        overlays.append(
            dl.Overlay(dl.LayerGroup(colorbar), checked=True, name="colorbar")
        )
        overlays.append(dl.Overlay(dl.LayerGroup(info), checked=True, name="info"))
        return dl.Map(
            dl.LayersControl([*DensityMap.get_dash_tilelayer(), *overlays]),
            center=(48.162, 11.586),
            zoom=18,
            style={
                "width": "100%",
                "height": "50vh",
                "margin": "auto",
                "display": "block",
            },
        )

    def init(self):
        print("init")


class _DashBoard:
    cell_err_app = CellErrorInsepctor
    map_view = _MapView


DashBoard = _DashBoard()
