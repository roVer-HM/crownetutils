from roveranalyzer.utils import PathHelper

ROOT = "/home/mkilian/repos/crownet/analysis/roveranalyzer/data"
RUN = "sumoBottleneck"

PAINT_INTERVALS = True
IS_VADERE_ANALYSIS = True if "vadere" in RUN else False
NODE_NAME = "node" if IS_VADERE_ANALYSIS else "pedestrianNode"

PATH_ROOT = f"{ROOT}/{RUN}"
p = PathHelper(PATH_ROOT)