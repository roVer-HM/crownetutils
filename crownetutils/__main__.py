import argparse
import sys

from crownetutils import __version__
from crownetutils.crownet_dash.flaskapp.wsgi import run_app_ns
from crownetutils.entrypoint.suqc_rerun_parser import append_suqc_rerun_parser


def parse_arguments():
    # parse arguments
    main: argparse.ArgumentParser = argparse.ArgumentParser(
        prog="Roveranalyzer", description=f"Version: {__version__}"
    )
    parent: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    # arguments used by all sub-commands
    # todo if needed
    main.add_argument(
        "-v", "--version", action="version", version="%(prog)s " + __version__
    )
    # subparsers
    sub = main.add_subparsers(title="Available Commands", dest="subparser_name")

    # Dash / Plotly
    dash_parser = sub.add_parser(
        "dash", help="Start Dash/Plotly Server for live analysis", parents=[parent]
    )
    dash_parser.add_argument("--suqc-dir", required=False, help="Suqc folder")
    dash_parser.add_argument("--run-cfg", required=False, help="Read run json config")
    dash_parser.add_argument(
        "--run-filter",
        "-f",
        required=False,
        default=None,
        help="Only load runs matching against string. Use preceding '!' to invert match",
    )
    dash_parser.set_defaults(main_func=run_app_ns)

    # Rerun postprocesing
    append_suqc_rerun_parser(sub, [parent])

    return main.parse_args()


if __name__ == "__main__":
    print("")
    ns: argparse.Namespace = parse_arguments()
    ret = ns.main_func(ns)
    if ret:
        sys.exit(0)
    else:
        sys.exit(-1)
