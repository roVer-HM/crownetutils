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
VALIDATE_RUN_COUNT = True


PAINT_INTERVALS = True
IS_VADERE_ANALYSIS = True if "vadere" in SIM_CONFIG else False


#
# output paths
#
OUT_PATH = f"{ROOT}/plots/{SIM_CONFIG}"
OUT_PATH_POSITION = f"{OUT_PATH}/position"
OUT_PATH_SPEED = f"{OUT_PATH}/speed"
OUT_PATH_DELAY = f"{OUT_PATH}/delay"
OUT_PATH_DISTANCE = f"{OUT_PATH}/distance"
OUT_PATH_PED_COUNT = f"{OUT_PATH}/ped_count"

OUT_PATHS_LIST = [OUT_PATH_POSITION, OUT_PATH_SPEED, OUT_PATH_DELAY, OUT_PATH_DISTANCE]  # , OUT_PATH_PED_COUNT]

for path in OUT_PATHS_LIST:
    PathHelper(path, create_missing=True)

