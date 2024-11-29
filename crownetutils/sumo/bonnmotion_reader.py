import contextlib
import gzip

import numpy as np
import numpy.linalg as lg
import pandas as pd


class BonnMotionReader:
    def __init__(self, path) -> None:
        self.path = path
        self.handler = None
        self.sumo_ids = None

    @contextlib.contextmanager
    def _open(self, path: str):
        fd = None
        try:
            if path.endswith(".gz"):
                fd = gzip.open(path, "rt", encoding="utf-8")
            else:
                fd = open(path, "rt", encoding="utf-8")
            yield fd
            fd.close()
            fd = None
        except Exception as e:
            if fd is not None:
                fd.close()
            raise e

    def __iter__(self):
        comment_count = 0
        ID_MAP_COMMENT = "# Sumo id map:"
        with self._open(self.path) as fd:
            for c, _row in enumerate(fd):
                if _row.startswith("#"):
                    comment_count += 1
                    if _row.startswith(ID_MAP_COMMENT):
                        _c = _row[len(ID_MAP_COMMENT) :].strip()
                        self.sumo_ids = np.array(_c.split(" ")).astype(int)
                    continue

                row_c = c - comment_count
                dbl_row = np.array(_row.split(" ")).astype(float)
                if len(dbl_row) % 3 != 0:
                    print(
                        f"row {row_c} number of elements in row are not divisible by 3, thus not a valid bonn motion trace. Skip."
                    )
                    continue
                if self.handler is not None:
                    dbl_row = self.handler(row_c, dbl_row)
                yield row_c, dbl_row

    def to_frame(self) -> pd.DataFrame:
        df = []
        for c, _row in self:
            # print(f"process row {c}")
            df.append(_row)
        df = pd.DataFrame(
            np.concatenate(df, axis=0),
            columns=["id", "time", "x", "y", "dist", "speed"],
        )
        df = df.fillna(0.0)
        df["id"] = df["id"].astype(int)
        return df


def row(id, data: np.array):
    """Parse bonnmotion row of the form ["id", "time", "x", "y", "dist", "speed"]"""
    data = data.reshape((-1, 3))
    data_shift = np.copy(data)
    data_shift[-1] = data_shift[0]
    data_shift = np.roll(data_shift, shift=1, axis=0)
    data_diff = data - data_shift
    dist = lg.norm(data_diff[:, 1:3], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = dist / data_diff[:, 0]
    data = np.concatenate(
        [
            np.repeat(id, data.shape[0]).reshape(-1, 1),
            data,
            dist.reshape(-1, 1),
            speed.reshape(-1, 1),
        ],
        axis=1,
    )
    return data


def bm_df_with_dist_and_speed(bm_file_path) -> pd.DataFrame:
    """Bonn motion traces in point form with columns
    ["id", "time", "x", "y", "dist", "speed"]
    """
    r = BonnMotionReader(bm_file_path)
    r.handler = row
    return r.to_frame(), r


def frame_from_bm(bm_file_path) -> pd.DataFrame:
    """Bonn motion traces in point form with columns
    ["id", "time", "x", "y", "dist", "speed"]
    """
    frame, _ = bm_df_with_dist_and_speed(bm_file_path)
    return frame
