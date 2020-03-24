import logging
import time


class Timer:
    @classmethod
    def create_and_start(cls, name, tabstop=0):
        return cls(name, tabstop)

    def __init__(self, name, tabstop):
        self._name = name
        self._tabstop = tabstop
        self._start = time.time()

    def stop(self):
        indet = "\t" * self._tabstop
        print(f"{indet}timer>> {time.time() - self._start:0.5f}s\t({self._name})")
        return self

    def start(self, name):
        self._name = name
        self._start = time.time()
        return self

    def stop_start(self, new_name):
        self.stop()
        self.start(new_name)
        return self
