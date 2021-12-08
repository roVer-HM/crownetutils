from roveranalyzer.utils import PathHelper

ROOT = "/home/mkilian/repos/crownet/analysis/roveranalyzer/data"
RUN = "sumoBottleneck"
# SPECIFIC_RUN = ""

PATH_ROOT = f"{ROOT}/{RUN}"
# PATH_SPECIFIC_RUN = f"{ROOT}/{RUN}/{SPECIFIC_RUN}"
p = PathHelper(PATH_ROOT)
# p_specific = PathHelper(PATH_SPECIFIC_RUN)

DENSITY_APP_INDEX = 0
BEACON_APP_INDEX = 1

PAINT_INTERVALS = True
IS_VADERE_ANALYSIS = True if "vadere" in RUN else False
NODE_NAME = "pNode"

PATH_ROOT = f"{ROOT}/{RUN}"
p = PathHelper(PATH_ROOT)
#
# position analysis settings:
#
OUTPUT_PATH = f"{PATH_ROOT}/out/position"
SPEED_OUTPUT_PATH = f"{PATH_ROOT}/out/speed"
DISTANCE_OUTPUT_PATH = f"{PATH_ROOT}/out/distance"
