import os

from roveranalyzer.utils import PathHelper

# constants
VADERE_SIMPLE = "vadereSimple"
VADERE_BOTTLENECK = "vadereBottleneck"
SUMO_SIMPLE = "sumoSimple"
SUMO_BOTTLENECK = "sumoBottleneck"
SIM = "cmp_vadere_sumo"
DENSITY_APP_INDEX = 0
BEACON_APP_INDEX = 1
NODE_NAME = "pNode"


ROOT = f"{os.environ['HOME']}/crownet/crownet/simulations/{SIM}/results"
SIM_CONFIG = VADERE_SIMPLE
RUN_COUNT = 10
VALIDATE_RUN_COUNT = False

# SPECIFIC_RUN = ""

# PATH_ROOT = f"{ROOT}/{SIM_CONFIG}"
PATH_ROOT = ROOT

# PATH_SPECIFIC_RUN = f"{ROOT}/{RUN}/{SPECIFIC_RUN}"
p = PathHelper(PATH_ROOT)
# p_specific = PathHelper(PATH_SPECIFIC_RUN)


PAINT_INTERVALS = True
IS_VADERE_ANALYSIS = True if "vadere" in SIM_CONFIG else False


#
# position analysis settings:
#
OUTPUT_PATH = f"{PATH_ROOT}/out/position"
SPEED_OUTPUT_PATH = f"{PATH_ROOT}/out/speed"
DISTANCE_OUTPUT_PATH = f"{PATH_ROOT}/out/distance"
