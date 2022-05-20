import argparse
from typing import Dict

from roveranalyzer.analysis.common import Simulation, SuqcRun
from roveranalyzer.analysis.flaskapp.application import init_app


def run_app_ns(ns: argparse.Namespace):
    run = SuqcRun(ns.suqc_dir)
    run_app(run.get_simulation_dict(lbl_key=True))


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
