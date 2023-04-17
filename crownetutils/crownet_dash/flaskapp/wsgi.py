import argparse
from typing import Dict

from crownetutils.analysis.common import RunMap, Simulation, SuqcStudy
from crownetutils.crownet_dash.flaskapp.application import init_app


def run_app_ns(ns: argparse.Namespace):
    runs = {}
    try:
        if ns.suqc_dir is not None:
            run = SuqcStudy(ns.suqc_dir)
            if ns.run_filter is not None:
                if ns.run_filter[0] == "!":
                    _f = lambda x: ns.run_filter not in x
                else:
                    _f = lambda x: ns.run_filter in x
                runs = {
                    k: v
                    for k, v in run.get_simulation_dict(lbl_key=True).items()
                    if _f(k)
                }
            else:
                runs = {k: v for k, v in run.get_simulation_dict(lbl_key=True).items()}
        elif ns.run_cfg is not None:
            run = RunMap.load_from_json(ns.run_cfg)
            for sim_g in run.values():
                for rep, sim in enumerate(sim_g):
                    runs.setdefault(f"{sim_g.lbl} ({rep})", sim)
    except Exception as e:
        print(
            f"\nCannot read {ns.suqc_dir} as suqc study directory (Error: {e}).\nTry to read single simulation result."
        )
        sim = Simulation.from_output_dir(ns.suqc_dir)
        runs = {sim.label: sim}

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
