import dash_bootstrap_components as dbc
from dash import Dash
from dash.dependencies import Input, Output, State
from flask import Flask

from roveranalyzer.analysis.flaskapp.application import cache
from roveranalyzer.analysis.flaskapp.application.layout import IdProvider, build_layout


def create_dashboard(server: Flask):
    dash_app = Dash(
        server=server,
        routes_pathname_prefix="/dash/",
        assets_folder="static",
        external_stylesheets=[dbc.themes.BOOTSTRAP],
    )

    build_layout(dash_app)

    init_callbacks(dash_app)

    return dash_app.server


def init_callbacks(app: Dash):

    sel_ids: IdProvider = app.sel_ids
    cell_err_ids: IdProvider = app.cell_err_ids
    scenario_ids: IdProvider = app.scenario_ids

    # @cache.memoize(timeout=10)
    # def time():
    #     return f"time: {datetime.datetime.now()}"

    @app.callback(sel_ids.out_("selector-dropdown", "options"), sel_ids.sig_in)
    def clb_data_selector(data):
        print(data)
        return ["a", "b", "c", "d", "f"]
