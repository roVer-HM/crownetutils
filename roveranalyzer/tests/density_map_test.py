import matplotlib.pyplot as plt
from oppanalyzer.utils import build_density_map

from uitls.path import PathHelper
from vadereanalyzer.plots.plots import t_cmap
from vadereanalyzer.plots.scenario import VaderScenarioPlotHelper

if __name__ == "__main__":
    s = PathHelper.rover_sim(
        "mucFreiNetdLTE2dMulticast/",
        "vadere00_geo_20200925-14:41:18/vadere.d/mf_2peds.scenario",
    ).abs_path()

    scenario = VaderScenarioPlotHelper(s)
    # read DataFrame. Use real coordinates and not the cell
    # count to allow background mesh
    path = PathHelper.rover_sim(
        "mucFreiNetdLTE2dMulticast/",
        "vadere00_geo_20200925-14:41:18/0a:aa:00:00:00:02.csv",
    ).abs_path()
    df = build_density_map(path, real_coords=True)
    X = df.index.levels[1].to_numpy()
    Y = df.index.levels[2].to_numpy()

    # create 2d array of density measures for pcolormesh
    dfs = df.loc[(1.0), ("count")].unstack().transpose()
    dfs2 = df.loc[(16.0), ("count")].unstack().transpose()
    dfs3 = df.loc[(28.0), ("count")].unstack().transpose()
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    for a in ax:
        for aa in a:
            aa.set_aspect("equal")
            aa.set_xlim([0, 100 * 3.0])
            aa.set_ylim([0, 80 * 3.0])
            scenario.add_obstacles(aa)

    # use transparent color map (transparent 0)
    cmap = t_cmap(cmap_name="Reds", replace_index=(0, 1, 0.0))
    ax[0, 0].pcolormesh(X, Y, dfs, cmap=cmap, edgecolor="k")
    ax[0, 0].set_title("t=1.0 (ID=9:1:268)")
    ax[0, 1].pcolormesh(X, Y, dfs2, cmap=cmap)
    ax[0, 1].set_title("t=26.0 (ID=9:1:268)")
    ax[1, 1].pcolormesh(X, Y, dfs3, cmap=cmap)
    ax[1, 1].set_title("t=50.0 (ID=9:1:268)")
    fig.show()
