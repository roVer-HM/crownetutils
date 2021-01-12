import os


def add_rover_env_var():
    os.environ["CROWNET_HOME"] = os.path.abspath("../../../")
    if os.environ["CROWNET_HOME"] is None:
        raise SystemError(
            "Please add CROWNET_HOME to your system variables to run a rover simulation."
        )
