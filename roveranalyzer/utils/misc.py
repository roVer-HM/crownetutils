import time

import numpy as np


def ccw(a, b, c):
    """
    is triangle abc counter clockwise?
    """
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def intersect(line1, line2):
    assert line1.shape == line2.shape == (2, 2)
    return ccw(line1[0], line2[0], line2[1]) != ccw(
        line1[1], line2[0], line2[1]
    ) and ccw(line1[0], line1[1], line2[0]) != ccw(line1[0], line1[1], line2[1])


class Timer:

    ACTIVE = True

    @classmethod
    def create_and_start(cls, name="timer", label=0):
        return cls(name, label)

    def __init__(self, name, label):
        self._name = name
        self._label = label
        self._start = time.time()

    def stop(self):

        if self.ACTIVE:
            print(
                f"{self._label}::timer>> {time.time() - self._start:0.5f}s\t({self._name})"
            )
        return self

    def start(self, name):
        self._name = name
        self._start = time.time()
        return self

    def stop_start(self, new_name):
        self.stop()
        self.start(new_name)
        return self


class StatsTool:
    """
    Toolset for calculating and nicely printing statistics
    """

    @staticmethod
    def stats_table(data, unit: str = "", name: str = "") -> str:
        """
        Create a table listing the most important statistic values

        :param data:    data to calculate the statistics on
        :param unit:    SI unit of data (optional)
        :param name:    name of the data to be printed (optional)
        :return:        string with statistics table
        """
        table = "=============================================================\n"
        if len(name) > 0:
            table += (
                f"! Data: {name:51} !\n"
                "-------------------------------------------------------------\n"
            )

        table += (
            f"! nr of values : {len(data):15}                            !\n"
            f"! arith. mean  : {np.mean(data):15.6f} {unit:>4}                       !\n"
            f"! minimum      : {np.min(data):15.6f} {unit:>4}                       !\n"
            f"! maximum      : {np.max(data):15.6f} {unit:>4}                       !\n"
            f"! median       : {np.median(data):15.6f} {unit:>4}                       !\n"
            f"! std. dev.    : {np.std(data):15.6f} {unit:>4}                       !\n"
            f"! variance     : {np.var(data):15.6f} {unit:>4}^2                     !\n"
            "=============================================================\n"
        )

        return table
