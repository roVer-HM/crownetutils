import os
import sys

import numpy as np

from roveranalyzer.simulators.vadere.plots.plots import DensityPlots

sys.path.append(
    os.path.abspath("")
)  # in case tutorial is called from the root directory
sys.path.append(os.path.abspath("../../../vadereanalyzer"))  # in tutorial directly


if __name__ == "__main__":

    directory = "testData/simple_detour"  # directory where following files are stored
    file_names = os.path.join(
        directory, "Mesh_trias1.txt"
    )  # from vadere data processor: MeshProcessor
    count_names = os.path.join(
        directory, "Density_trias1.csv"
    )  # count_names = from vadere data processor: MeshDensityCountingProcessor

    # plot (smoothed) density or counts for one frame /one timestep
    frame = 200  # time = frame* 0.4
    density_plots = DensityPlots(directory, file_names, count_names)
    density_plots.plot_density(frame, "counts")
    density_plots.plot_density(frame, "density")
    density_plots.plot_density(frame, "density_smooth")

    # plot (smoothed) density or counts for time series
    frames = np.arange(390, 400, 1)  # frames, time = 0.4* frame, min = 1!
    density_plots.animate_density(frames, "counts", "counts_movie")
    density_plots.animate_density(frames, "density", "density_movie")
    density_plots.animate_density(frames, "density_smooth", "density_smooth_movie")
