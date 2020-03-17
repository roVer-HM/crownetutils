import pandas as pd


def num_pedestrians_time_series(df, ax, title, c_start, c_end, c_count, ret_data=False):
    """
    creates time series plot of number of pedestrians in the simulation based on
    the 'endTime' and 'startTime' processors.

    returns axis with plot and copy of DataFrame if ret_data is true.
    """
    df_in = df.loc[:, c_count].groupby(df[c_start]).count()
    if type(df_in) == pd.Series:
        df_in = df_in.to_frame()
    df_in = df_in.rename({c_count: "in"}, axis=1)

    df_out = df.loc[:, c_count].groupby(df[c_end]).count()
    df_out = df_out.to_frame()
    df_out = df_out.rename({c_count: "out"}, axis=1)
    df_io = pd.merge(df_in, df_out, how="outer", left_index=True, right_index=True)
    df_io = df_io.fillna(0)
    df_io["in_cum"] = df_io["in"].cumsum()
    df_io["out_cum"] = df_io["out"].cumsum()
    df_io["diff_cum"] = df_io["in_cum"] - df_io["out_cum"]

    ax.scatter(df_io.index, df_io["diff_cum"], marker=".", linewidths=0.15)
    ax.set_title(f"{title} -#Peds")
    ax.set_ylabel("number of Peds")
    ax.set_xlabel("simulation time [s]")

    if ret_data:
        return ax, df_io.copy(deep=True)
    else:
        return ax
