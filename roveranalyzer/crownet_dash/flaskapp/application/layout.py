from __future__ import annotations

from typing import Any

import dash_bootstrap_components as dbc
from dash import Dash, dash_table, dcc, html
from dash.dependencies import Input, Output, State
from dash_extensions.javascript import Namespace

from roveranalyzer.crownet_dash.dashapp import DashUtil


class IdProvider:
    def __init__(self, prefix, suffix="") -> None:
        self.prefix = prefix
        self.suffix = suffix
        self.keys = set()
        self._signal = dcc.Store(self("signal"))

    def __call__(self, var: str, *args: Any, **kwds: Any) -> Any:
        self.keys.add(var)
        return f"{self.prefix}-{var}-{self.suffix}"

    def out_(self, var, opt, prefix="") -> Output:
        return Output(f"{prefix}{self(var)}", opt)

    def in_(self, var, opt, prefix="") -> Input:
        return Input(f"{prefix}{self(var)}", opt)

    def state(self, var, opt, prefix="") -> State:
        return State(f"{prefix}{self(var)}", opt)

    def insert_layout_defaults(self, layout):
        return html.Div(
            id=self("container"),
            className="module-wrapper",
            children=[self.signal, *layout],
        )

    @property
    def sig_out(self):
        return self.out_("signal", "data")

    @property
    def sig_in(self):
        return self.in_("signal", "data")

    @property
    def sig_state(self):
        return self.state("signal", "data")

    @property
    def signal(self):
        return self._signal


def build_layout(dash_app: Dash):
    ns = Namespace("dashExtensions", "default")
    selector_layout, selector_ids = data_selector_view()
    cell_error_layout, cell_error_ids = cell_errors_view()
    scenario_layout, scenario_ids = scenario_view(ns)
    misc_layout, misc_ids = misc_view(ns)

    ns.dump(assets_folder=dash_app.config.assets_folder)

    dash_app.layout = dbc.Container(
        [
            selector_layout,
            misc_layout,
            scenario_layout,
            cell_error_layout,
        ]
    )

    dash_app.selector_ids = selector_ids
    dash_app.cell_err_ids = cell_error_ids
    dash_app.scenario_ids = scenario_ids
    dash_app.misc_ids = misc_ids
    return dash_app


def full_row(content):
    return dbc.Row([dbc.Col([content])])


def full_graph(id):
    return full_row(dcc.Graph(id=id))


def tbl_row(id):
    return full_row(
        dash_table.DataTable(
            id=id,
            page_action="none",
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            fixed_rows={"headers": True},
            style_table=dict(height="600px", overflowY="auto", overflowX="scroll"),
        )
    )


def slider_btn(id):
    return dbc.Row(
        [
            dbc.Col(
                width=1,
                children=[
                    html.Button("<", id=f"t_back-{id}", n_clicks=0),
                    html.Button(">", id=f"t_forward-{id}", n_clicks=0),
                ],
            ),
            dbc.Col(
                width=11,
                children=[
                    dcc.Slider(
                        id=id,
                        step=None,
                        marks={0: {"label": "0s"}},
                        included=False,
                        tooltip={"placement": "bottom", "always_visible": True},
                    )
                ],
            ),
        ],
        className="ctrl",
        align="center",
    )


def lbl_dropdown(label, lbl_width, dp_id, dp_args: dict = {}, **kwargs):
    dp_args.setdefault("multi", False)
    dp_args.setdefault("searchable", True)
    kwargs.setdefault("align", "center")
    kwargs.setdefault("className", "ctrl")

    return dbc.Row(
        [
            dbc.Col(width=lbl_width, children=[label]),
            dbc.Col(width=12 - lbl_width, children=[dcc.Dropdown(id=dp_id, **dp_args)]),
        ],
        **kwargs,
    )


def misc_view(ns: Namespace, ids: IdProvider | None = None):
    if ids is None:
        ids = IdProvider("misc-view")

    layout = [
        DashUtil.module_header(ids("title")),
        lbl_dropdown("Select Plot:", 3, ids("plot-dropdown")),
        lbl_dropdown("Select Node:", 3, ids("node-dropdown")),
        full_row(html.Div(id=ids("plot-wrapper"), children=["No plot selected"])),
    ]

    return ids.insert_layout_defaults(layout), ids


def scenario_view(ns: Namespace, ids: IdProvider | None = None):
    if ids is None:
        ids = IdProvider("map-view")

    layout = [
        DashUtil.module_header(ids("title")),
        lbl_dropdown("Select Node:", 3, ids("node-dropdown")),
        full_row(
            dcc.RadioItems(
                ["count", "err", "sqerr"],
                "count",
                className="ctrl",
                labelClassName="ctrl-lbl",
                inline=True,
                id=ids("map-view-function"),
            )
        ),
        full_row(DashUtil.map_view(ns=ns, id_builder=ids)),
        slider_btn(ids("time-slider")),
        dcc.Store(id=ids("hover-store")),
    ]

    return ids.insert_layout_defaults(layout), ids


def cell_errors_view(ids: IdProvider | None = None):
    if ids is None:
        ids = IdProvider("cell-err")

    layout = [
        DashUtil.module_header(id=ids("title"), children=["Cell Error View"]),
        dbc.Row(
            dbc.Col(
                html.Button(
                    id=ids("load"),
                    children="Load",
                    style={"width": "100%", "height": "100%"},
                )
            )
        ),
        lbl_dropdown("Select Cell:", 3, ids("cell-dropdown")),
        lbl_dropdown("Select Id:", 3, ids("node-id-dropdown")),
        tbl_row(ids("cell-tbl")),
        full_graph(id=ids("cell-err-graph")),
        full_graph(id=ids("measurement-count-graph")),
        full_graph(id=ids("cell-beacon-graph")),
        #
        full_row(html.H4(id=ids("cell-measurement-header"))),
        tbl_row(ids("cell-measurement-tbl")),
        slider_btn(ids("time-slider")),
        #
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id=ids("cell-measurement-hist-time"))),
                dbc.Col(dcc.Graph(id=ids("cell-measurement-hist-dist"))),
            ]
        ),
    ]

    return ids.insert_layout_defaults(layout), ids


def data_selector_view(ids: IdProvider | None = None):
    if ids is None:
        ids = IdProvider("data-selector")

    layout = [
        full_row(
            dcc.Dropdown(id=ids("selector-dropdown"), multi=False, searchable=True)
        ),
        full_row(
            html.Iframe(
                id=ids("pdf-view"),
                src="",
                style={"visibility": "hidden", "width": "100%", "height": "600px"},
            )
        ),
        full_row(
            dash_table.DataTable(
                id=ids("config-tbl"),
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
        ),
    ]

    return ids.insert_layout_defaults(layout), ids
