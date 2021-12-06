import re

import pandas as pd
from matplotlib import pyplot as plt

from roveranalyzer.utils.file import read_lines


def cumulative_messages(
    df,
    ax=None,
    msg=("msg_present", "msg_in_fes"),
    lbl=("number of messages", "messages in fes"),
    set_lbl=True,
):
    fig = None
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for idx, m in enumerate(msg):
        ax.plot("time", m, data=df, label=lbl[idx])

    if set_lbl:
        ax.set_xlabel("time [s]")
        ax.set_ylabel("number of messages")
        ax.legend()
        ax.set_title("messages in simulation")

    if fig is None:
        return ax
    else:
        return fig, ax


def parse_cmdEnv_outout(path):
    lines = read_lines(path)

    pattern1 = re.compile(
        "^\*\* Event #(?P<event>\d+)\s+t=(?P<time>\S+)\s+Elapsed: (?P<elapsed>\S+?)s\s+\((?P<elapsed_s>.*?)\).*?completed\s+\((?P<completed>.*?)\% total\)"
    )
    pattern2 = re.compile(
        "^.*?Speed:\s+ev/sec=(?P<events_per_sec>\S+)\s+simsec/sec=(?P<simsec_per_sec>\S+)\s+ev/simsec=(?P<elapsed>\S+)"
    )
    pattern3 = re.compile(
        "^.*?Messages:\s+created:\s+(?P<msg_created>\d+)\s+present:\s+(?P<msg_present>\d+)\s+in\s+FES:\s+(?P<msg_in_fes>\d+)"
    )

    data = []
    event_data = []
    for l in lines:
        if l.strip().startswith("** "):
            if len(event_data) != 0:
                data.append(event_data)
            event_data = []
            if m := pattern1.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        elif l.strip().startswith("Speed:"):
            if m := pattern2.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        elif l.strip().startswith("Messages:"):
            if m := pattern3.match(l):
                event_data.extend(list(m.groups()))
            else:
                raise ValueError("ddd")
        else:
            break

    col = list(pattern1.groupindex.keys())
    col.extend(list(pattern2.groupindex.keys()))
    col.extend(list(pattern3.groupindex.keys()))
    df = pd.DataFrame(data, columns=col)
    df = df.apply(pd.to_numeric, errors="ignore")
    return df
