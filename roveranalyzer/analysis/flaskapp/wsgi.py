import argparse
from typing import Dict

from roveranalyzer.analysis.common import Simulation, SuqcRun
from roveranalyzer.analysis.flaskapp.application import init_app


def run_app_ns(ns: argparse.Namespace):
    run = SuqcRun(ns.suqc_dir)
    if ns.run_filter is not None:
        if ns.run_filter[0] == "!":
            _f = lambda x: ns.run_filter not in x
        else:
            _f = lambda x: ns.run_filter in x
        runs = {k: v for k, v in run.get_simulation_dict(lbl_key=True).items() if _f(k)}
    if runs == {}:
        raise ValueError(f"No run selected with filter: {ns.run_filter}")
    run_app(runs)


def run_app(simulations: Dict[str, Simulation]):
    app = init_app(simulations)
    print("run Flask app !!")
    app.run(host="127.0.0.1", port=5051, debug=True, use_reloader=False)


if __name__ == "__main__":
    sims = {
        "sim1": Simulation(
            "/mnt/data1tb/results/ymfDistDbg2/simulation_runs/outputs/Sample_0_0/final_out/",
            "sim1",
        ),
        "sim2": Simulation(
            "/mnt/data1tb/results/ymfDistDbg2/simulation_runs/outputs/Sample_1_0/final_out/",
            "sim2",
        ),
    }
    run_app(sims)
