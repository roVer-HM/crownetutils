import os

from roveranalyzer.utils import PathHelper

SIM = "cmp_vadere_sumo"
SIM_CONFIG = "vadereSimple"
RUN = ""
RUN_COUNT = 10
ROOT = f"{os.environ['HOME']}/crownet/crownet/simulations/{SIM}/results"

# SPECIFIC_RUN = ""

# PATH_ROOT = f"{ROOT}/{SIM_CONFIG}"
PATH_ROOT = ROOT

# PATH_SPECIFIC_RUN = f"{ROOT}/{RUN}/{SPECIFIC_RUN}"
p = PathHelper(PATH_ROOT)
# p_specific = PathHelper(PATH_SPECIFIC_RUN)

DENSITY_APP_INDEX = 0
BEACON_APP_INDEX = 1

PAINT_INTERVALS = True
IS_VADERE_ANALYSIS = True if "vadere" in SIM_CONFIG else False
NODE_NAME = "pNode"

VALIDATE_RUN_COUNT = False

#
# position analysis settings:
#
OUTPUT_PATH = f"{PATH_ROOT}/out/position"
SPEED_OUTPUT_PATH = f"{PATH_ROOT}/out/speed"
DISTANCE_OUTPUT_PATH = f"{PATH_ROOT}/out/distance"
