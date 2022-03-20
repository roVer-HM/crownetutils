import copy
from os import getsid
from os.path import join
from re import I
from typing import Dict

import dash_bootstrap_components as dbc
import plotly.express as px
import rasterio
from dash import Dash, callback_context, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from flask import Flask
from numpy import sign

import roveranalyzer.analysis.flaskapp.application.model as m
from roveranalyzer.analysis.dashapp import DashUtil
from roveranalyzer.analysis.flaskapp.application import cache
from roveranalyzer.analysis.flaskapp.application.layout import IdProvider, build_layout
from roveranalyzer.analysis.omnetpp import OppAnalysis
from roveranalyzer.utils.logging import timing


def create_dashboard(server: Flask):
    dash_app = Dash(
        server=server,
        routes_pathname_prefix="/dash/",
        assets_folder="static",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    print(f"first: {m.get_count_index.cache_info()}")
    sims = {
        "sim1": m.Simulation(
            "/mnt/data1tb/results/ymfDistDbg2/simulation_runs/outputs/Sample_0_0/final_out/",
            "sim1",
        ),
        "sim2": m.Simulation(
            "/mnt/data1tb/results/ymfDistDbg2/simulation_runs/outputs/Sample_1_0/final_out/",
            "sim2",
        ),
    }
    for sim in sims.values():
        sim.copy_pdf(
            "common_output.pdf", sim, join(dash_app.config.assets_folder, "pdf")
        )

    build_layout(dash_app)

    init_callbacks(dash_app, sims)

    return dash_app.server


def init_callbacks(app: Dash, sims: Dict[str, m.Simulation]):

    selector_ids: IdProvider = app.selector_ids
    cell_err_ids: IdProvider = app.cell_err_ids
    scenario_ids: IdProvider = app.scenario_ids

    def get_sim(signal):
        if signal is None or "sim" not in signal:
            raise ValueError("Sim not set")
        return sims[signal["sim"]]

    @app.callback(
        selector_ids.out_("selector-dropdown", "options"), selector_ids.sig_in
    )
    def simulation_option_clb(data):
        if data is not None:
            raise PreventUpdate()
        ret = list(sims.keys())
        ret.sort()
        return ret

    @app.callback(
        selector_ids.sig_out,
        selector_ids.in_("selector-dropdown", "value"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def select_simulation_clb(sim_selection, signal):
        signal = signal or {}
        signal["sim"] = sim_selection
        # todo start thread to load stuff
        return signal

    @app.callback(
        selector_ids.out_("pdf-view", "src"),
        selector_ids.out_("pdf-view", "style"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def select_pdf(signal):
        if signal and "sim" in signal:
            print(f"signal: {signal}")
            sim = signal["sim"]
            style = {"width": "100%", "height": "600px"}
            return sims[sim].asset_pdf_path("common_output", suffix=sim), style
        raise ValueError("not set")

    @app.callback(
        selector_ids.out_("config-tbl", "data"),
        selector_ids.out_("config-tbl", "columns"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def select_config_table(signal):
        if signal and "sim" in signal:
            print(f"signal: {signal}")
            sim = sims[signal["sim"]]
            df = sim.sql.get_all_run_config()

            return df.to_dict("records"), DashUtil.cols(df.columns)
        raise ValueError("not set")

    def set_time_slider_marks(signal):
        sim = get_sim(signal)
        t_index = m.get_time_index(sim)
        opt = {k: {"label": f"{k}s", "style": {"display": "none"}} for k in t_index}
        for k in range(0, t_index.shape[0], 10):
            t = t_index[k]
            del opt[t]["style"]["display"]
        return opt

    app.callback(
        scenario_ids.out_("time-slider", "marks"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )(set_time_slider_marks)

    app.callback(
        cell_err_ids.out_("time-slider", "marks"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )(set_time_slider_marks)

    # multiple sliders will register here
    def time_slider_btn_click(slider, backwards, forwards, signal, *args, **kwargs):
        time_index = m.get_time_index(get_sim(signal))
        trigger = callback_context.triggered[0]["prop_id"]
        if slider is None:
            return time_index[0]
        if "t_back-" in trigger and slider > time_index[0]:
            i = time_index.to_list().index(slider)
            return i
        elif "t_forward-" in trigger and slider < time_index[-1]:
            i = time_index.to_list().index(slider)
            return i + 2
        else:
            raise PreventUpdate("")

    app.callback(
        scenario_ids.out_("time-slider", "value"),
        scenario_ids.in_("time-slider", "value"),
        scenario_ids.in_("time-slider", "n_clicks", prefix="t_back-"),
        scenario_ids.in_("time-slider", "n_clicks", prefix="t_forward-"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )(time_slider_btn_click)
    app.callback(
        cell_err_ids.out_("time-slider", "value"),
        cell_err_ids.in_("time-slider", "value"),
        cell_err_ids.in_("time-slider", "n_clicks", prefix="t_back-"),
        cell_err_ids.in_("time-slider", "n_clicks", prefix="t_forward-"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )(time_slider_btn_click)

    @timing
    @app.callback(
        scenario_ids.out_("node-dropdown", "value"),
        scenario_ids.out_("node-dropdown", "options"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def set_map_view(signal):
        sim = get_sim(signal)
        host_ids = m.get_host_ids(sim)
        opt = [{"value": k, "label": f"{k} - {v}"} for k, v in host_ids.items()]
        return opt[0]["value"], opt

    @app.callback(
        scenario_ids.out_("cell_tiles", "hideout"),
        scenario_ids.out_("map-colorbar-wrapper", "children"),
        scenario_ids.in_("map-view-function", "value"),
        prevent_initial_call=True,
    )
    def set_map_view_function(view_f):
        classes, colorscale, colorbar, style = DashUtil.get_colorbar(
            id=scenario_ids("map-colorbar"), value=view_f
        )
        return (
            dict(
                colorscale=colorscale,
                classes=classes,
                style=style,
                colorProp=view_f,
            ),
            colorbar,
        )

    @app.callback(
        scenario_ids.out_("topo_tiles", "data"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def set_topography(signal):
        sim = get_sim(signal)
        return m.get_topography_json(sim)

    @app.callback(
        scenario_ids.out_("cell_tiles", "data"),
        scenario_ids.out_("node_tiles", "data"),
        scenario_ids.in_("time-slider", "value"),
        scenario_ids.in_("node-dropdown", "value"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def update_table_view(time, node, signal):
        if any(i is None for i in [time, node, signal]):
            print("None Input!!")
            raise PreventUpdate
        sim = get_sim(signal)
        return (
            m.get_cell_tile_geojson_for(sim, time, node),
            m.get_node_tile_geojson_for(sim, time, node),
        )

    @app.callback(
        scenario_ids.out_("title", "children"),
        scenario_ids.in_("node-dropdown", "value"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def update_map_view_title(node, signal):
        value = 0 if node is None else node
        host_id = m.get_host_ids(get_sim(signal)).get(value, "???")
        return f"Density Map for Node {value} - {host_id}"

    @app.callback(
        scenario_ids.out_("cell_info", "children"),
        scenario_ids.out_("hover-store", "data"),
        scenario_ids.in_("cell_tiles", "hover_feature"),
        scenario_ids.state("hover-store", "data"),
        prevent_initial_call=True,
    )
    def update_map_view_info(feature, storage):
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

    @app.callback(
        cell_err_ids.out_("cell-tbl", "data"),
        cell_err_ids.out_("cell-tbl", "columns"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def set_erroneous_cells(signal):
        sim = get_sim(signal)
        df = m.get_erroneous_cells(sim)
        cols = [dict(name=i, id=i) for i in df.columns]
        return df.to_dict("records"), cols

    @app.callback(
        cell_err_ids.out_("cell-dropdown", "options"),
        cell_err_ids.out_("cell-dropdown", "value"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def set_cell_dropdown(signal):
        sim = get_sim(signal)
        cells = m.get_cells(sim)
        cell_map = {k: tuple(v) for k, v in enumerate(cells)}
        opt = [{"value": k, "label": f"[{v[0]},{v[1]}]"} for k, v in cell_map.items()]
        return opt, opt[0]["value"]

    @app.callback(
        cell_err_ids.out_("node-id-dropdown", "options"),
        cell_err_ids.out_("node-id-dropdown", "value"),
        cell_err_ids.in_("cell-dropdown", "value"),
        selector_ids.sig_in,
        prevent_initial_call=True,
    )
    def set_id_dropdown(cell_id, signal):
        sim = get_sim(signal)
        cell_id = 0 if cell_id is None else cell_id
        ids = m.get_node_ids_for_cell(sim, cell_id)
        host_ids = m.get_host_ids(sim)
        opt = [{"value": int(k), "label": f"{k} - {host_ids[k]}"} for k in ids[1:]]
        return opt, opt[0]["value"]

    @app.callback(
        cell_err_ids.out_("cell-err-graph", "figure"),
        cell_err_ids.in_("node-id-dropdown", "value"),
        cell_err_ids.in_("cell-dropdown", "value"),
        selector_ids.sig_state,
        prevent_initial_call=True,
    )
    def update_cell_error_graph(node_id, cell_id, signal):
        sim = get_sim(signal)
        cell_id = 0 if cell_id is None else cell_id
        ca = m.get_cell_error_data(sim, cell_id)

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
        time_index = m.get_time_index(sim)
        fig.update_xaxes(range=[time_index.min(), time_index.max()])
        return fig

    @app.callback(
        cell_err_ids.out_("cell-beacon-graph", "figure"),
        cell_err_ids.in_("node-id-dropdown", "value"),
        cell_err_ids.in_("cell-dropdown", "value"),
        selector_ids.sig_state,
        prevent_initial_call=True,
    )
    def update_beacon_figure(node_id, cell_index, signal):
        sim = get_sim(signal)
        cell = m.get_cells(sim)[cell_index]
        df = m.get_beacon_entry_exit(sim, node_id, cell)
        fig = px.scatter(
            df.reset_index(),
            x="event_time",
            y="cell_change_cumsum",
            title=f"Cell occupancy based on beacons from node {node_id} for cell {cell}",
            custom_data=[df["source_node"]],
        )
        time_index = m.get_time_index(sim)
        fig.update_xaxes(range=[time_index.min(), time_index.max()])
        fig.update_layout(hovermode="x unified")
        fig.update_traces(
            mode="markers+lines",
            hovertemplate="value: %{y}</b> source: %{customdata[0]}",
        )
        return fig

    @app.callback(
        cell_err_ids.out_("measurement-count-graph", "figure"),
        cell_err_ids.in_("node-id-dropdown", "value"),
        cell_err_ids.in_("cell-dropdown", "value"),
        selector_ids.sig_state,
        prevent_initial_call=True,
    )
    def update_measurement_count_figure(node_id, cell_id, signal):
        sim = get_sim(signal)
        df = m.get_measurement_count_df(sim, node_id, cell_id)

        fig = px.scatter(
            df,
            x="simtime",
            y="count",
            title=f"Number of measuremtens from which to choose",
            # hover_data=["count"],
        )
        time_index = m.get_time_index(sim)
        fig.update_xaxes(range=[time_index.min(), time_index.max()])
        fig.update_layout(hovermode="x unified")
        fig.update_traces(
            mode="markers+lines",
            hovertemplate="count: %{y}",
        )
        return fig

    @app.callback(
        cell_err_ids.out_("cell-measurement-tbl", "data"),
        cell_err_ids.out_("cell-measurement-tbl", "columns"),
        cell_err_ids.out_("cell-measurement-tbl", "tooltip_header"),
        cell_err_ids.out_("title", "children"),
        cell_err_ids.out_("cell-measurement-hist-time", "figure"),
        cell_err_ids.out_("cell-measurement-hist-dist", "figure"),
        cell_err_ids.in_("time-slider", "value"),
        cell_err_ids.state("node-id-dropdown", "value"),
        cell_err_ids.state("cell-dropdown", "value"),
        selector_ids.sig_state,
        prevent_initial_call=True,
    )
    def update_measurements_tbl(time, node_id, cell_id, signal):
        if any([i is None for i in [time, node_id, cell_id]]):
            raise PreventUpdate
        sim = get_sim(signal)
        df = m.get_measurements(sim, time, node_id, cell_id)

        cols = DashUtil.cols(
            df.columns, ignore=["delay", "update_age", "x_owner", "y_owner"]
        )
        cols.insert(0, dict(id="source", name="source"))
        records = (
            df.reset_index(["simtime", "x", "y", "ID"], drop=True)
            .reset_index()
            .to_dict("records")
        )

        x, y = m.get_cells(sim)[cell_id]
        h3 = f"Density measurements for cell [{x},{y}] for time {time} from point of view of {node_id}-{m.get_host_ids(sim)[node_id]}"

        hist1 = px.histogram(df, x="ymfD_t", title="age factor")
        hist2 = px.histogram(df, x="ymfD_d", title="distance factor")

        return records, cols, {i["id"]: i["id"] for i in cols}, h3, hist1, hist2
