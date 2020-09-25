import matplotlib.pyplot as plt
from oppanalyzer.utils import build_density_map

from uitls.path import PathHelper

if __name__ == "__main__":
    path = PathHelper.rover_sim(
        "mucFreiNetdLTE2dMulticast/",
        "vadere00_geo_20200925-14:41:18/0a:aa:00:00:00:02.csv",
    ).abs_path()
    df = build_density_map(path)
    dfs = df.loc[(1.0), ("count")].unstack().transpose()
    dfs2 = df.loc[(16.0), ("count")].unstack().transpose()
    dfs3 = df.loc[(28.0), ("count")].unstack().transpose()
    fig, ax = plt.subplots(2, 2, figsize=(16, 16))
    ax[0, 0].pcolor(dfs, cmap="RdBu")
    ax[0, 0].set_title("t=1.0 (ID=9:1:268)")
    ax[0, 1].pcolor(dfs2, cmap="RdBu")
    ax[0, 1].set_title("t=26.0 (ID=9:1:268)")
    ax[1, 1].pcolor(dfs3, cmap="RdBu")
    ax[1, 1].set_title("t=50.0 (ID=9:1:268)")
    for a in ax:
        for aa in a:
            aa.set_aspect("equal")
            aa.set_xlim([0, 80])
            aa.set_ylim([0, 80])
    fig.show()
