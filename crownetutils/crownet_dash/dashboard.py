from __future__ import annotations

import uuid
from typing import List

import dash_bootstrap_components as dbc
import dash_leaflet as dl
import numpy as np
import pandas as pd
import plotly.express as px
from dash import callback_context, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from dash_extensions.javascript import Namespace

from crownetutils.crownet_dash.dashapp import DashUtil, OppModel, dash_app
from crownetutils.crownet_dash.density_map import DensityMap
from crownetutils.utils.logging import timing


class _DashApp:
    def __init__(self, name, id_prefix, m: OppModel) -> None:
        self.name = name
        self.id_prefix = id_prefix
        # self.app = Dash(__name__)
        self.fig = None
        self.m = m

    def get_layout(self, ns: Namespace, with_container: bool = False):
        pass

    def init(self):
        pass

    def run_app(self, ns: Namespace | None):
        ns = Namespace("dashExtensions", "default") if ns is None else ns
        self.init()
        dash_app.layout = self.get_layout(ns, with_container=True)
        ns.dump(assets_folder=dash_app.config.assets_folder)
        dash_app.run_server(debug=True, use_reloader=False)

    def id(self, name, attr=None):
        if attr is None:
            return f"{self.id_prefix}_{name}"
        else:
            return f"{self.id_prefix}_{name}.{attr}"


class CellErrorInsepctor(_DashApp):
    def __init__(self, m: OppModel) -> None:
        super().__init__("Cell Error Inspector", "cell", m)
        self.selected_node_id = None

        dash_app.callback(
            Output(self.id("cell-node-id-dropdown"), "options"),
            Output(self.id("cell-node-id-dropdown"), "value"),
            Input(self.id("cell-dropdown"), "value"),
            Input("session-id", "data"),
            Input("data-selector", "value"),
            # prevent_initial_call=True
        )(self.clb_update_id_dropdown)

        dash_app.callback(
            Output(self.id("cell-tbl"), "data"),
            Output(self.id("cell-tbl"), "columns"),
            Input("data-selector", "value"),
        )(self.clb_tbl)

        dash_app.callback(
            Output(self.id("cell-err-graph"), "figure"),
            Input(self.id("cell-node-id-dropdown"), "value"),
            Input("session-id", "data"),
            prevent_initial_call=True,
        )(self.clb_update_error_figure)

        dash_app.callback(
            Output(self.id("cell-beacon-graph"), "figure"),
            Input(self.id("cell-node-id-dropdown"), "value"),
            State(self.id("cell-dropdown"), "value"),
            prevent_initial_call=True,
        )(self.clb_update_beacon_figure)

        dash_app.callback(
            Output(self.id("measurement-count-graph"), "figure"),
            Input(self.id("cell-node-id-dropdown"), "value"),
            State(self.id("cell-dropdown"), "value"),
            prevent_initial_call=True,
        )(self.clb_update_measurement_count)

        dash_app.callback(
            Output(self.id("cell-dropdown"), "options"),
            Input(("data-selector"), "value"),
        )(self.clb_update_cell_dropdown_options)

        dash_app.callback(
            Output(self.id("cell-measurements-tbl"), "data"),
            Output(self.id("cell-measurements-tbl"), "columns"),
            Output(self.id("cell-measurements-tbl"), "tooltip_header"),
            Output(self.id("cell-measurement-header"), "children"),
            Output(self.id("cell-measurement-hist-time"), "figure"),
            Output(self.id("cell-measurement-hist-dist"), "figure"),
            Input(self.id("measurement-time-slider"), "value"),
            Input(self.id("cell-node-id-dropdown"), "value"),
            Input(self.id("cell-dropdown"), "value"),
            prevent_initial_call=True,
        )(self.clb_update_measurement_table)
        # input node-id, cell, time (slider)
        # output table of measurements present at time

        dash_app.callback(
            Output(self.id("measurement-time-slider"), "marks"),
            Input("data-selector", "value"),
        )(self.clb_update_time_slider_option)

    def clb_update_cell_dropdown_options(self, data):
        self.m.data_changed(data)
        cell_map = {k: tuple(v) for k, v in enumerate(self.m.cells)}
        return [{"value": k, "label": f"[{v[0]},{v[1]}]"} for k, v in cell_map.items()]

    def clb_update_time_slider_option(self, data):
        self.m.data_changed(data)
        opt = {
            k: {"label": f"{k}s", "style": {"display": "none"}}
            for k in self.m.map_time_index
        }
        for k in range(0, self.m.map_time_index.shape[0], 10):
            t = self.m.map_time_index[k]
            del opt[t]["style"]["display"]
        return opt

    def clb_update_measurement_count(self, node_id, cell_id):
        x, y = self.m.cells[cell_id]
        df = self.m.builder.map_p[
            pd.IndexSlice[:, float(x), float(y), :, node_id], ["count"]
        ]
        df = df.groupby(by=["simtime", "x", "y"]).count().reset_index()
        fig = px.scatter(
            df,
            x="simtime",
            y="count",
            title=f"Number of measuremtens from which to choose",
            # hover_data=["count"],
        )
        fig.update_xaxes(
            range=[self.m.map_time_index.min(), self.m.map_time_index.max()]
        )
        fig.update_layout(hovermode="x unified")
        fig.update_traces(
            mode="markers+lines",
            hovertemplate="count: %{y}",
        )
        return fig

    def clb_update_measurement_table(self, time, node_id, cell_id):
        x, y = self.m.cells[cell_id]
        # df = self.m.builder.map_p[
        #     pd.IndexSlice[float(time), float(x), float(y), :, node_id], :
        # ]

        # dist_sum = df["sourceEntry"].sum()
        # time_sum = df["measurement_age"].sum()
        # alpha = 0.5

        # t_min = df["measurement_age"].min()
        # time_sum_new = df["measurement_age"].sum() - df["measurement_age"].shape[0]*t_min

        # df["ymfD_t"] = alpha*((df["measurement_age"] - t_min)/time_sum_new)

        # df["ymfD_d"] = (1-alpha)* (df["sourceEntry"]/dist_sum)
        # df["ymfD"] = df["ymfD_t"] + df["ymfD_d"]
        df = self.m.get_measurements(time, node_id, cell_id)

        cols = DashUtil.cols(
            df.columns, ignore=["delay", "update_age", "x_owner", "y_owner"]
        )
        cols.insert(0, dict(id="source", name="source"))

        h3 = f"Density measurements for cell [{x},{y}] for time {time} from point of view of {self.m.host_ids[node_id]}"

        records = (
            df.reset_index(["simtime", "x", "y", "ID"], drop=True)
            .reset_index()
            .to_dict("records")
        )

        hist1 = px.histogram(df, x="ymfD_t", title="age factor")
        hist2 = px.histogram(df, x="ymfD_d", title="distance factor")
        return records, cols, {i["id"]: i["id"] for i in cols}, h3, hist1, hist2

    def clb_tbl(self, data):
        self.m.data_changed(data)
        df = self.m.erroneous_cells.iloc[0:30]
        cols = [dict(name=i, id=i) for i in df.columns]
        return df.to_dict("records"), cols

    def get_layout(self, ns: Namespace, with_container: bool = False):
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
                    className="ctrl",
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
                    className="ctrl",
                ),
                dbc.Row(
                    dbc.Col(
                        dash_table.DataTable(
                            id=self.id("cell-tbl"),
                            page_action="none",
                            filter_action="native",
                            sort_action="native",
                            sort_mode="multi",
                            fixed_rows={"headers": True},
                            style_table=dict(height="300px", overflowY="auto"),
                        )
                    )
                ),
                dbc.Row(dbc.Col(dcc.Graph(id=self.id("cell-err-graph")))),
                dbc.Row(dbc.Col(dcc.Graph(id=self.id("measurement-count-graph")))),
                dbc.Row(dbc.Col(dcc.Graph(id=self.id("cell-beacon-graph")))),
                *DashUtil.build_measurement_tbl(
                    self.m,
                    self.id("cell-measurements-tbl"),
                    self.id("measurement-time-slider"),
                    self.id("cell-measurement-header"),
                ),
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(id=self.id("cell-measurement-hist-time"))),
                        dbc.Col(dcc.Graph(id=self.id("cell-measurement-hist-dist"))),
                    ]
                ),
            ],
            id=self.id("wrapper"),
            className="module-wrapper",
        )
        if with_container:
            layout = dbc.Container(layout)
        return layout

    # callback
    @timing
    def clb_update_error_figure(self, node_id, session):
        ca = self.m.ctx.get(session, "cell_error", None)
        _mask = ca["ID"] == 0
        if node_id is not None:
            node_id = [node_id] if isinstance(node_id, int) else node_id
            for v in node_id:
                _mask = _mask | (ca["ID"] == v)

        fig = px.scatter(
            ca[_mask],
            x="simtime",
            y="count",
            color="id",
            title=f"Count error for selected cell based on density map data",
            hover_data=["id", "count", "err"],
        )
        fig.update_xaxes(
            range=[self.m.map_time_index.min(), self.m.map_time_index.max()]
        )
        return fig

    @timing
    def clb_update_beacon_figure(self, node_id, cell_index):
        cell = self.m.cells[cell_index]
        # df = pd.DataFrame(self.m.map_time_index).rename(columns={"simtime": "event_time"})
        # df["cell_change_cumsum"] = 0
        df = self.m.get_beacon_entry_exit(node_id, cell)
        fig = px.scatter(
            df.reset_index(),
            x="event_time",
            y="cell_change_cumsum",
            title=f"Cell occupancy based on beacons from node {node_id} for cell {cell}",
            custom_data=[df["source_node"]],
        )
        fig.update_xaxes(
            range=[self.m.map_time_index.min(), self.m.map_time_index.max()]
        )
        fig.update_layout(hovermode="x unified")
        fig.update_traces(
            mode="markers+lines",
            hovertemplate="value: %{y}</b> source: %{customdata[0]}",
        )
        return fig

    # callback
    @timing
    def clb_update_id_dropdown(self, cell_id, session, data):
        self.m.data_changed(data)
        ids = self.read_ids_for_cell_index(cell_id, session)["ID"].unique()
        # ids = sself.ca["ID"].unique()
        ids = ids[ids > 0]

        return [
            {"value": int(k), "label": f"{k} - {self.m.host_ids[k]}"} for k in ids
        ], ids[0]

    @timing
    def read_ids_for_cell_index(self, cell_id, session):
        i = pd.IndexSlice
        cell_id = 0 if cell_id is None else cell_id
        _cell = self.m.cells[cell_id]
        with self.m.builder.count_p.query as ctx:
            # all nodes, all times given _cell
            ca = ctx.select(
                key=self.m.builder.count_p.group,
                where=f"(x == {float(_cell[0])}) & (y == {float(_cell[1])})",
            )
        ca = ca.reset_index(["x", "y"]).sort_index()

        # only ground truth for given cell
        ca_0 = ca.loc[i[:, 0], :]
        # create missing [time, ID] index for ground truth (i.e. np.zeros(..))
        idx_diff = pd.Index(
            zip(self.m.map_time_index, np.zeros(self.m.map_time_index.shape[0]))
        ).difference(ca_0.index)

        # append missing counts to data frame...
        _add_zero = pd.DataFrame(index=idx_diff, columns=ca_0.columns).fillna(0)
        ca = pd.concat([ca, _add_zero])
        ca = ca.reset_index()
        ca["id"] = ca["ID"].astype("str")
        self.m.ctx.set(session, "cell_error", ca)
        return ca

    def init(self):
        self.cwerr = self.m.erroneous_cells
        # self.read_ids_for_cell_index(cell_id=None)  # todo...


class _DataSelector(_DashApp):
    def __init__(self, m: OppModel, data) -> None:
        super().__init__("Data Selector", "data", m)
        self.data = data
        dash_app.callback(
            Output(self.id("pdf-view"), "src"),
            Input("data-selector", "value"),
            Input("data-selector", "options"),
            # prevent_initial_call=True
        )(self.clb_pdf)
        dash_app.callback(
            Output(self.id("config-tbl"), "data"),
            Output(self.id("config-tbl"), "columns"),
            Input("data-selector", "value"),
        )(self.clb_config_tbl)

    def clb_config_tbl(self, data):
        self.m.data_changed(data)
        df = self.m.sql.get_all_run_config()
        df = df[df["configKey"] == "*.pNode[*].app[1].app.mapCfg"]
        return df.to_dict("records"), DashUtil.cols(df.columns)

    def clb_pdf(self, data, options):
        for o in options:
            if data == o["value"]:
                p = self.m.asset_pdf_path(data, relative=True, suffix=o["label"])
                print(p)
                return p

    def get_layout(self, ns: Namespace, with_container: bool = False):
        layout = html.Div(
            children=[
                dcc.Store(id=self.id("memory")),
                dbc.Row(
                    [
                        dbc.Col(
                            DashUtil.data_dropdown("data-selector", self.m, self.data)
                        )
                    ],
                    align="center",
                    className="ctrl",
                ),
                dbc.Row(
                    html.Iframe(
                        # To my recollection you need to put your static files in the 'assets' folder
                        id=self.id("pdf-view"),
                        src="assets/example.pdf",
                        style={"width": "100%", "height": "600px"},
                    ),
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dash_table.DataTable(
                                id=self.id("config-tbl"),
                                page_action="none",
                                filter_action="native",
                                sort_action="native",
                                sort_mode="multi",
                                fixed_rows={"headers": True},
                                style_cell=dict(textAlgin="left"),
                                style_cell_conditional=[
                                    {"if": {"column_id": c}, "textAlign": "left"}
                                    for c in ["configKey", "configValue"]
                                ],
                                style_table=dict(height="300px", overflowY="auto"),
                            )
                        )
                    ]
                ),
            ],
            id=self.id("wrapper"),
            className="module-wrapper",
        )
        if with_container:
            layout = dbc.Container(layout)

        return layout


class _MapView(_DashApp):
    def __init__(self, m: OppModel) -> None:
        super().__init__("Map View", "map", m)

        # ### callbacks
        dash_app.callback(
            Output(self.id("cell_tiles"), "data"),
            Output(self.id("node_tiles"), "data"),
            Input(self.id("time_slider"), "value"),
            Input(self.id("map_node_input"), "value"),
            Input("data-selector", "value"),
            # prevent_initial_call=True
        )(self.clb_map_time_slider)
        dash_app.callback(
            Output(self.id("topo_tiles"), "data"),
            Input("data-selector", "value"),
            #    prevent_initial_call=True
        )(self.clb_topo)
        dash_app.callback(
            Output(self.id("cell_tiles"), "hideout"),
            Output(self.id("map-colorbar-wrapper"), "children"),
            Input(self.id("map-value-view-function"), "value"),
            prevent_initial_call=True,
        )(self.clb_cell_tile_update_colorbar)

        dash_app.callback(
            Output(self.id("time_slider"), "value"),
            Input(self.id("time_slider"), "value"),
            Input(self.id("time-back"), "n_clicks"),
            Input(self.id("time-forward"), "n_clicks"),
            prevent_initial_call=True,
        )(self.clb_time_click)

        dash_app.callback(
            Output(self.id("time_slider"), "marks"), Input("data-selector", "value")
        )(self.clb_update_time_slider_option)

        dash_app.callback(
            Output(self.id("map_title"), "children"),
            Input(self.id("map_node_input"), "value"),
            Input("session-id", "data"),
            Input("data-selector", "value"),
            # prevent_initial_call=True
        )(self.clb_map_node_input)
        dash_app.callback(
            Output(self.id("cell_info"), "children"),
            Output(self.id("memory"), "data"),
            Input(self.id("cell_tiles"), "hover_feature"),
            State(self.id("memory"), "data"),
            prevent_initial_call=True,
        )(DashUtil.get_cell_info)

    def clb_update_time_slider_option(self, data):
        self.m.data_changed(data)
        opt = {
            k: {"label": f"{k}s", "style": {"display": "none"}}
            for k in self.m.map_time_index
        }
        for k in range(0, self.m.map_time_index.shape[0], 10):
            t = self.m.map_time_index[k]
            del opt[t]["style"]["display"]
        return opt

    def clb_cell_tile_update_colorbar(self, color_function):
        # todo make colorbar dynamic?
        classes, colorscale, colorbar, style = DashUtil.get_colorbar(
            id=self.id("map-colorbar"), value=color_function
        )
        return (
            dict(
                colorscale=colorscale,
                classes=classes,
                style=style,
                colorProp=color_function,
            ),
            colorbar,
        )

    @timing
    def clb_map_time_slider(self, time_value, node_id, data):
        self.m.data_changed(data)
        return (
            self.m.get_cell_tile_geojson_for(time_value, node_id),
            self.m.get_node_tile_geojson_for(time_value, node_id),
        )

    @timing
    def clb_topo(self, data):
        self.m.data_changed(data)
        return self.m.topo_json()

    @timing
    def clb_map_time_input(self, value):
        return value

    @timing
    def clb_map_node_input(self, value, session_id, data):
        self.m.data_changed(data)
        value = 0 if value is None else value
        self.m.set_selected_node(session_id, value)
        print(f"selected node {value}")
        return f"Density Map for Node {value} - {self.m.host_ids.get(value, '???')}"

    @timing
    def clb_bar(self, value):
        # print(f"got drag value: {value}")
        return value

    def clb_time_click(self, slider, backward, forward):
        trigger = callback_context.triggered[0]["prop_id"]
        if (
            trigger == self.id("time-back", "n_clicks")
            and slider > self.m.map_time_index[0]
        ):
            i = self.m.map_time_index.to_list().index(slider)
            return i
        elif (
            trigger == self.id("time-forward", "n_clicks")
            and slider < self.m.map_time_index[-1]
        ):
            i = self.m.map_time_index.to_list().index(slider)
            return i + 2
        else:
            raise PreventUpdate("")

    def get_layout(self, ns: Namespace, with_container: bool = False):
        layout = html.Div(
            children=[
                dcc.Store(id=self.id("memory")),
                DashUtil.module_header(self.id("map_title")),
                dbc.Row(
                    [
                        dbc.Col(width=3, children=[html.Div("Select Node:")]),
                        dbc.Col(
                            DashUtil.id_dropdown(self.id("map_node_input"), self.m)
                        ),
                    ],
                    align="center",
                    className="ctrl",
                ),
                dbc.Row(
                    dbc.Col(
                        dcc.RadioItems(
                            ["count", "err", "sqerr"],
                            "count",
                            className="ctrl",
                            labelClassName="ctrl-lbl",
                            inline=True,
                            id=self.id("map-value-view-function"),
                        )
                    )
                ),
                dbc.Row(dbc.Col([DashUtil.map_view(ns=ns, id_builder=self.id)])),
                dbc.Row(
                    [
                        dbc.Col(
                            width=1,
                            children=[
                                html.Button("<", id=self.id("time-back"), n_clicks=0),
                                html.Button(
                                    ">", id=self.id("time-forward"), n_clicks=0
                                ),
                            ],
                        ),
                        dbc.Col(
                            DashUtil.build_time_slider(self.m, self.id("time_slider"))
                        ),
                    ],
                    className="ctrl",
                    align="center",
                ),
            ],
            id=self.id("wrapper"),
            className="module-wrapper",
            # style={
            #     "width": "80%",
            #     "margin": "auto",
            #     "display": "block",
            #     "position": "relative",
            # },
        )
        if with_container:
            layout = dbc.Container(layout)

        return layout

    def _build_map_layout(self, ns: Namespace, id_builder):
        topo_overlays = DashUtil.build_topography_layer(ns, id_builder)
        cell_overlays = DashUtil.build_cell_layer(ns, id_builder)
        node_overlays = DashUtil.build_node_layer(ns, id_builder)

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
                "height": "50vh",
                "background-color": "white",
                "margin": "auto",
                "display": "block",
            },
        )


class _Combined(_DashApp):
    def __init__(self, name, id_prefix, m: OppModel) -> None:
        super().__init__(name, id_prefix, m)
        self.components: List[_DashApp] = []

    def add(self, _cls, *args):
        self.components.append(_cls(self.m, *args))

    def run_app(self, ns: Namespace | None = None):
        def layout():
            ns = Namespace("dashExtensions", "default")
            _id = uuid.uuid4()
            print(f"build layout {_id}")
            layout = []
            for c in self.components:
                c.init()
                layout.append(c.get_layout(ns, with_container=False))
            layout = html.Div([dcc.Store(data=str(_id), id="session-id"), *layout])
            layout = dbc.Container(layout)
            ns.dump(assets_folder=dash_app.config.assets_folder)
            return layout

        print("add layout callback")
        dash_app.layout = layout()
        dash_app.run_server(debug=True, use_reloader=False)


class _DashBoard:
    cell_err_app = CellErrorInsepctor
    map_view = _MapView
    combined = _Combined
    data_selector = _DataSelector


DashBoard = _DashBoard()
