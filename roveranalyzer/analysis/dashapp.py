import collections
import json
from re import I

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import dash_leaflet.express as dlx
import pandas as pd
from dash import Dash, callback_context, dash_table, dcc, html
from dash.dependencies import Input, Output
from dash_extensions.javascript import Namespace, arrow_function

import roveranalyzer.simulators.opp as OMNeT
from roveranalyzer.simulators.crownet.dcd.dcd_builder import DcdHdfBuilder
from roveranalyzer.utils.general import Project

dash_app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
dash_app.config.suppress_callback_exceptions = True
i_ = pd.IndexSlice


class OppModel:
    def __init__(
        self, data_root: str, builder: DcdHdfBuilder, sql: OMNeT.CrownetSql
    ) -> None:
        self.data_root: str = data_root
        self.builder: DcdHdfBuilder = builder
        self.sql: OMNeT.CrownetSql = sql

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
        self.selected_node = self.map_node_index[0]  # global
        self.map_data = None
        self.map_time_cache = None

    def set_selected_node(self, value):
        if self.selected_node != value:
            self.selected_node = value
            return True
        return False

    def get_geojson_for(self, time_value):
        print("check")
        if (
            self.map_data is None
            or self.map_time_cache is None
            or time_value not in self.map_time_cache
        ):
            print(f"load from hdf")
            self.map_data = self.builder.count_p.geo(Project.WSG84_lat_lon)[
                pd.IndexSlice[
                    time_value - 3 : time_value + 3, :, :, int(self.selected_node)
                ]
            ]
            self.map_time_cache = (
                self.map_data.index.get_level_values("simtime").unique().sort_values()
            )
        j = json.loads(self.map_data.loc[time_value].reset_index().to_json())
        print(f"geojson for {time_value}")
        return j

    @property
    def erroneous_cells(self):
        if hasattr(self, "_erroneous_cells"):
            return self._erroneous_cells

        with self.builder.count_p.query as ctx:
            self._erroneous_cells = ctx.select(
                key=self.builder.count_p.group,
                where="(ID>0) & (err > 0)",
                columns=["count", "err"],
            )

        _mask = self._erroneous_cells["count"] == self._erroneous_cells["err"]
        self._erroneous_cells = self._erroneous_cells[_mask]
        self._erroneous_cells = (
            self._erroneous_cells.groupby(by=["x", "y"])
            .sum()
            .sort_values("count", ascending=False)
            .reset_index()
        )


class _DashUtil:
    @classmethod
    def get_colorbar(cls, width=300, height=30, position="bottomleft", **kwargs):
        classes = [0, 1, 2, 3, 4, 5, 6, 7]
        colorscale = [
            "#FFEDA0",
            "#FED976",
            "#FEB24C",
            "#FD8D3C",
            "#FC4E2A",
            "#E31A1C",
            "#BD0026",
            "#800026",
        ]
        # Create colorbar.
        ctg = [
            "{}".format(cls, classes[i + 1]) for i, cls in enumerate(classes[:-1])
        ] + ["{}+".format(classes[-1])]
        colorbar = dlx.categorical_colorbar(
            categories=ctg,
            colorscale=colorscale,
            width=width,
            height=height,
            position=position,
            **kwargs,
        )
        return classes, colorscale, colorbar

    _colorbar_style_clb = """function(feature, context){
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
        }
    """

    @classmethod
    def get_map_style(cls, ns: Namespace):
        style = dict(weight=2, opacity=1, fillColor="red", fillOpacity=0.6)
        clb = ns.add(cls._colorbar_style_clb)
        return style, clb

    @classmethod
    def module_header(cls, id, **kwargs):
        return dbc.Row(dbc.Col([html.H2(id=id, **kwargs)]))

    # callback
    def build_map(self, value=0):
        ns = Namespace("dashExtensions", "default")
        classes, colorscale, colorbar = DashUtil.get_colorbar()
        # Create info control.
        info = html.Div(
            id="cell-info",
            className="info",
            style={
                "position": "absolute",
                "bottom": "100px",
                "left": "10px",
                "z-index": "1000",
            },
        )

        style = dict(weight=2, opacity=1, fillColor="red", fillOpacity=0.6)
        style_handle = ns.add(
            """function(feature, context){
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
        }"""
        )
        ns.dump(assets_folder=dash_app.config.assets_folder)

        self.update_map_node(value)
        self.map_time_index = self.cells.index.get_level_values("simtime").unique()
        print("split data")
        self.map_geojson = json.loads(
            self.cells.loc[self.map_time_index[0]].reset_index().to_json()
        )
        print("geojson created")
        overlays = []
        geoj = dl.GeoJSON(
            options=dict(style=ns(style_handle)),
            # options=dict(style=arrow_function(style)),
            data=self.map_geojson,
            zoomToBounds=True,
            zoomToBoundsOnClick=True,
            format="geojson",
            hoverStyle=arrow_function(dict(weight=3, color="#222", dashArray="")),
            hideout=dict(
                colorscale=colorscale, classes=classes, style=style, colorProp="count"
            ),
            id="cells",
        )

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
        )

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
    def get_cell_info(cls, feature=None):
        header = [html.H4("Cell Info")]
        if not feature:
            return header + [html.P("Hoover over a cell")]
        else:
            ret = header + [
                html.B(
                    f"[{feature['properties']['cell_x']}, {feature['properties']['cell_y']}]"
                ),
                html.Br(),
                f"Count: {feature['properties']['count']}",
                html.Br(),
                f"Error: {feature['properties']['err']}",
                html.Br(),
                f"Owner Distance: {feature['properties']['owner_dist']:0.3f} m",
            ]
            return ret

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


DashUtil = _DashUtil()
