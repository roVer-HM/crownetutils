import os


def add_rover_env_var():
    os.environ["ROVER_MAIN"] = os.path.abspath("../../../")
    if os.environ["ROVER_MAIN"] is None:
        raise SystemError(
            "Please add ROVER_MAIN to your system variables to run a rover simulation."
        )
