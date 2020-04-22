import logging
import time


class Timer:

    ACTIVE = True

    @classmethod
    def create_and_start(cls, name, label=0):
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
