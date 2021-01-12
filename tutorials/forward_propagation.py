#!/usr/bin/env python3
#!/usr/bin/python3

import matplotlib.pyplot as plt
from SALib.analyze.sobol import analyze
from SALib.sample import saltelli
from suqc import *
from utils.general import add_rover_env_var

# sys.path.append(os.path.abspath("."))
# sys.path.append(os.path.abspath(".."))


###############################################################################################################


def qoi():
    ## Define the quantities of interest (simulation output variables)
    # Make sure that corresponding post processing methods exist in the run_script2.py file
    qoi = [
        "degree_informed_extract.txt",
        "poisson_parameter.txt",
        "time_95_informed.txt",
    ]
    return qoi


def problem_definition():
    problem = {
        "num_vars": 3,
        "names": [
            "number_of_agents_mean",
            "*.hostMobile[*].app[1].messageLength",
            "**wlan[*].radio.transmitter.power",
        ],
        "bounds": [[50, 100], [0, 50], [0.5, 2.0]],  # uniform distribution assumed!
    }
    return problem


def get_sampling_df(nr_samples=160):

    # STEP 1: Create samples with Sobol sequence using SALib
    parameter = problem_definition()

    param_values = saltelli.sample(
        parameter, int(nr_samples / 8), calc_second_order=True, seed=111
    )

    # Step 2: Make samples readable for suqc
    param_values = pd.DataFrame(
        param_values, columns=["number_of_agents_mean", "p1", "p2"]
    )
    param_values["number_of_agents_mean"] = round(
        param_values["number_of_agents_mean"], 0
    )

    # Step 2.1: Distribute number of agents at four sources and determine number of agents/(1 second)
    for x in [1, 2, 5, 6]:
        param_values[f"sources.[id=={x}].distributionParameters"] = param_values.apply(
            lambda row: [row.number_of_agents_mean * 0.01 / 4], axis=1
        )

    # Step 2.2: Add units
    param_values["*.hostMobile[*].app[1].messageLength"] = param_values.apply(
        lambda row: f"{int(row.p1)}B", axis=1
    )
    param_values["**wlan[*].radio.transmitter.power"] = param_values.apply(
        lambda row: f"{round(row.p2,2)}mW", axis=1
    )

    param_values = param_values.drop(columns=["p1", "p2"])
    return param_values


def get_sampling(nr_samples=2000, is_test=False):

    param_values = get_sampling_df(nr_samples=nr_samples)

    if is_test:
        param_values["number_of_agents_mean"] = 30
        param_values = param_values.iloc[np.linspace(0, 4), :]

    par_var = list()

    # Step 2.3: Create dictionary from dataframe which can be read by the suqc
    for x in range(len(param_values)):
        r = param_values.iloc[x].values
        d = {
            "dummy": {"number_of_agents_mean": r[0]},
            "vadere": {
                "sources.[id==1].distributionParameters": r[1],
                "sources.[id==2].distributionParameters": r[2],
                "sources.[id==5].distributionParameters": r[3],
                "sources.[id==6].distributionParameters": r[4],
            },
            "omnet": {
                "*.hostMobile[*].app[1].messageLength": r[5],
                "**wlan[*].radio.transmitter.power": r[6],
            },
        }
        par_var.append(d)

    if nr_samples != len(par_var):
        print(
            f"WARNING: The number of required sampled is {nr_samples}. {len(par_var)} were produced."
        )

    return par_var


def path2ini():

    path2ini = os.path.join(
        os.environ["CROWNET_HOME"],
        "rover/simulations/simple_detoure_suqc_traffic/omnetpp.ini",
    )
    return path2ini


def save_results(par_var, data, output_folder_name="output_df"):

    summary = os.path.join(os.getcwd(), output_folder_name)
    if os.path.exists(summary):
        shutil.rmtree(summary)

    os.makedirs(summary)

    par_var.to_csv(os.path.join(summary, "metainfo.csv"))

    data["poisson_parameter.txt"].to_csv(os.path.join(summary, "poisson_parameter.csv"))
    data["degree_informed_extract.txt"].to_csv(
        os.path.join(summary, "degree_informed_extract.csv")
    )
    data["time_95_informed.txt"].to_csv(os.path.join(summary, "time_95_informed.csv"))


def run_rover_simulations(par_var, result_dir="output_df"):

    add_rover_env_var()

    setup = CoupledDictVariation(
        ini_path=path2ini(),
        config="final",
        parameter_dict_list=par_var,
        qoi=qoi(),
        model=CoupledConsoleWrapper(model="Coupled"),
        scenario_runs=1,
        post_changes=PostScenarioChangesBase(apply_default=True),
        output_path=os.getcwd(),
        output_folder=os.path.join(os.getcwd(), "output"),
        remove_output=True,
        seed_config={"vadere": "fixed", "omnet": "fixed"},
        env_remote=None,
    )

    par_var, data = setup.run(-1)
    save_results(par_var, data, output_folder_name=result_dir)

    return data


def read_data(summary):

    # Check data
    parameter = pd.read_csv(
        os.path.join(summary, "metainfo.csv"), index_col=["id", "run_id"]
    )

    # extract data, remove units
    parameter = parameter.iloc[:, 0:3]
    parameter.columns = [c_name.split("'")[5] for c_name in parameter.columns.to_list()]
    print("Extracted parameters:")
    for col in parameter.columns.to_list():
        print(f"\tParameter: {col}")
        try:
            parameter[col] = (
                parameter[col].str.extract(r"(\d+(\.\d+)?)").astype("float")
            )
        except:
            pass

    dissemination_time = pd.read_csv(
        os.path.join(summary, "time_95_informed.csv"), index_col=[0, 1]
    )
    dissemination_time = dissemination_time[["timeToInform95PercentAgents"]]
    dissemination_time = dissemination_time.sort_index()
    return parameter, dissemination_time


if __name__ == "__main__":

    # Forward propagation and sensitivity analysis

    ## Step 1: Generate parameter combinations
    # the following 3 parameters are varied

    # 1) number_of_agents_mean:
    ## the number of agents generated in 100s
    ## lower bound: 50, upper bound: 100
    ## distribution: uniform

    # 2) *.hostMobile[*].app[1].messageLength
    ## is used to vary the network load, traffic load = messageLength*20ms
    ## lower bound: 0B, upper bound: 50B (video streaming in medium quality)
    ## distribution: uniform

    # 3) **wlan[*].radio.transmitter.power
    ## is used to define the transmitter power of the mobile devices
    ## lower bound: 0.50mW, upper bound: 2.00mW
    ## distribution: uniform

    nr_parameter_combinations = 10
    par_var = get_sampling(nr_parameter_combinations)

    ## STEP 2: Run simulations and store results in results dir

    results = "results"
    data = run_rover_simulations(par_var=par_var, result_dir=results)

    ## Step 3: Forward propagation: analyze distribution of dissemination time
    # parameter, dissemination_time = read_data(results)
    dissemination_time = data["time_95_informed.txt"]
    plt.hist(dissemination_time)
    plt.show()
    print(dissemination_time.describe())

    ## Step 4: Sensitivity analysis
    Si = analyze(
        problem=problem_definition(),
        Y=dissemination_time,
        calc_second_order=True,
        print_to_console=True,
    )

    "Forward propagation and sensitivity analysis finished."
