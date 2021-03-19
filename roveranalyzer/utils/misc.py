import logging
import time


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
