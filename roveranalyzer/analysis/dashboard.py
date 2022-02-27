from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.simulators.crownet.dcd.dcd_builder import DcdHdfBuilder
from roveranalyzer.utils.general import Project
from dash import html, dcc, dash_table, callback_context
import plotly.express as px
from dash.dependencies import Input, Output
import dash_leaflet as dl
import dash_leaflet.express as dlx
from dash_extensions.javascript import arrow_function, Namespace
import json

from roveranalyzer.analysis.density_map import DensityMap
from roveranalyzer.analysis.dashapp import dash_app

class _DashApp:
    def __init__(self, name) -> None:
        self.name = name
        # self.app = Dash(__name__)
        self.fig = None

    def build_layout(self):
        pass

    def init():
        pass

    def run_app(self):
        self.init()
        self.build_layout()
        dash_app.run_server(debug=True, use_reloader=False)


class CellErrorInsepctor(_DashApp):
    def __init__(self, data_root: str, builder: DcdHdfBuilder, sql: OMNeT.CrownetSql) -> None:
        super().__init__("Cell Error Inspector")
        self.data_root: str = data_root
        self.b: DcdHdfBuilder = builder
        self.s: OMNeT.CrownetSql = sql

        self.cells = self.b.count_p.geo(Project.WSG84_lat_lon)[pd.IndexSlice[325., :, :, 0]]
        self.nodes = self.s.host_position(
            epsg_code_base=Project.UTM_32N,
            # epsg_code_to=Project.OpenStreetMaps,
            epsg_code_to=Project.WSG84_lat_lon,
            time_slice=slice(324.8),
        )

        dash_app.callback(
            Output("id-dropdown", "options"),
            [Input("cell-dropdown", "value")],
        )(self.update_id_dropdown)

        dash_app.callback(
            Output("cell-info", "children"),
            [Input("cells", "hover_feature")]
        )(self.get_map_info)

        dash_app.callback(
            Output("cell-err-graph", "figure"),
            [Input("id-dropdown", "value")],
        )(self.update_error_figure)




    def build_layout(self):


        dash_app.layout = html.Div(children=[
            html.H1(children=f"{self.name}"),

            html.Div(children='''
                Show cell specifc difference of cell count (density) for one node. Node 0 show the ground truth.
            '''),
            html.Div(children=[
                dash_table.DataTable(self.cwerr.iloc[0:15].to_dict("records"), columns=[{"name": i, "id": i} for i in self.cwerr.columns])
                ], #children
                style={'width': '80%', 'margin': "auto", "display": "block", "position": "relative"}
            ),
            self.cell_dropdown,
            dcc.Graph(id='cell-err-graph',),
            self.id_dropdown,
            html.H2(children="Map View"),
            html.Div( children=[
                self.build_map()
                ], # children
                id="map",
                style={'width': '80%', 'height': '50vh', 'margin': "auto", "display": "block", "position": "relative"}
            ),
            # html.Div(children="",
            #     style={'width': '100%', 'height': '50vh', 'margin': "auto", "display": "block", "position": "relative"}
            # )
        ])


    def build_map(self):
        ns = Namespace("dashExtensions", "default")
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        colorscale = ['#FFEDA0', '#FED976', '#FEB24C', '#FD8D3C', '#FC4E2A', '#E31A1C', '#BD0026', '#800026']
        # Create colorbar.
        ctg = ["{}".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])] + ["{}+".format(classes[-1])]
        colorbar = dlx.categorical_colorbar(categories=ctg, colorscale=colorscale, width=300, height=30, position="bottomleft")
        # Create info control.
        info = html.Div(id="cell-info", className="info",
                style={"position": "absolute", "bottom": "100px", "left": "10px", "z-index": "1000"})

        style = dict(weight=2, opacity=1, fillColor='red', fillOpacity=0.6)
        style_handle = ns.add("""function(feature, context){
            const {classes, colorscale, style, colorProp} = context.props.hideout;  // get props from hideout
            const value = feature.properties[colorProp];  // get value the determines the color
            if (value < classes.length){
                style.fillColor = colorscale[value];
                style.color = colorscale[value];
            } else {
                style.fillColor = colorscale[classes.length];
                style.color = colorscale[classes.length];
            }
            return style;
        }""")
        ns.dump(assets_folder=dash_app.config.assets_folder)


        overlays = []
        geoj = dl.GeoJSON(
            options=dict(style=ns(style_handle)),
            # options=dict(style=arrow_function(style)),
            data=json.loads(self.cells.reset_index().to_json()),
            zoomToBounds=True,
            zoomToBoundsOnClick=True,
            format="geojson",
            hoverStyle=arrow_function(dict(weight=3, color='#222', dashArray='')),
            hideout=dict(colorscale=colorscale, classes=classes, style=style, colorProp="count"),
            id="cells"
        )

        overlays.append(
            dl.Overlay(
                dl.LayerGroup(
                    geoj
                ),
                checked=True,
                name="cells",
            )
        )
        overlays.append(
            dl.Overlay(
                dl.LayerGroup(
                    colorbar
                ),
                checked=True,
                name="colorbar"
            )
        )
        overlays.append(
            dl.Overlay(
                dl.LayerGroup(
                    info
                ),
                checked=True,
                name="info"
            )
        )
        return dl.Map(
            dl.LayersControl([*DensityMap.get_dash_tilelayer(),  *overlays]),
            center=(48.162, 11.586),
            zoom=18
        )

    def get_map_info(self, feature=None):
        header = [html.H4("Cell Info")]
        if not feature:
            return header + [html.P("Hoover over a cell")]
        return header + [html.B(f"[{feature['properties']['cell_x']}, {feature['properties']['cell_y']}]"),
                        html.Br(),
                        f"Count: {feature['properties']['count']}"]

    def update_error_figure(self, value):
        _mask = (self.ca["ID"] == 0)
        if value is not None:
            value = [value] if isinstance(value, int) else value
            for v in value:

                _mask = _mask | (self.ca["ID"] == v)

        fig = px.scatter(self.ca[_mask],
            x="simtime",
            y="count",
            color="id",
            hover_data=["id", "count", "err"],
            )
        fig.update_xaxes(range = [self.time_index.min(),self.time_index.max()])
        return fig

    def update_id_dropdown(self, value, *args, **kwargs):
        self.read_ids_for_cell_index(value)
        ids = self.ca["ID"].unique()
        ids = ids[ids > 0]
        return [{"value": int(i), "label": str(i)} for i in ids]

    def read_ids_for_cell_index(self, value, *args, **kwargs):
        i = pd.IndexSlice
        value = 0 if value is None else value
        _v = self.cell_map[value]
        with self.b.count_p.query as ctx:
            ca = ctx.select(key=self.b.count_p.group, where=f"(x == {float(_v[0])}) & (y == {float(_v[1])})")
        self.ca = ca.reset_index(["x", "y"]).sort_index()

        ca_0 = self.ca.loc[i[:, 0], :]
        idx_diff = pd.Index(zip(self.time_index, np.zeros(self.time_index.shape[0]))).difference(ca_0.index)
        _add_zero = pd.DataFrame(index=idx_diff, columns=ca_0.columns).fillna(0)
        self.ca = pd.concat([self.ca, _add_zero])
        self.ca = self.ca.reset_index()
        self.ca["id"] = self.ca["ID"].astype("str")


    def init(self):
        with self.b.count_p.query as ctx:
            # all time points present
            self.time_index = ctx.select(key=self.b.count_p.group, where="(ID==0)", columns=[])
            self.time_index = self.time_index.index.get_level_values("simtime").unique().sort_values()

            cells = ctx.select(key=self.b.count_p.group, columns=["x", "y"])
            cells = cells.reset_index(["simtime", "ID"], drop=True).sort_index().index.unique()
            self.cwerr = ctx.select(key=self.b.count_p.group, where="(ID>0) & (err > 0)", columns=["count","err"])

        _mask = self.cwerr["count"] ==  self.cwerr["err"]
        self.cwerr = self.cwerr[_mask]
        self.cwerr = self.cwerr.groupby(by=["x", "y"]).sum().sort_values("count", ascending=False).reset_index()

        self.cell_map = {k: tuple(v) for k, v in enumerate(cells)}
        self.cell_dropdown = dcc.Dropdown(
            id="cell-dropdown",
            options=[{"value": k, "label": f"[{v[0]},{v[1]}]"} for k,v in self.cell_map.items()],
            multi=False,
            searchable=True,
            value=0
        )
        self.read_ids_for_cell_index(0)
        ids = self.ca["ID"].unique()
        ids = ids[ids > 0]
        options = [{"value": int(i), "label": str(i)} for i in ids]
        self.id_dropdown = dcc.Dropdown(
            id="id-dropdown",
            options = options,
            multi=True,
            searchable=True,
            value = options[0]["value"]
        )


class _DashBoard:
    cell_err_app = CellErrorInsepctor

DashBoard = _DashBoard()