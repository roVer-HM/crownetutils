import logging
import os
import timeit as it
from functools import update_wrapper, wraps
from typing import Any, List


class NoConnectionPoolFilter(logging.Filter):
    def filter(self, record):
        return not (record.module == "connectionpool" and record.levelname == "DEBUG")


# logging.root.addFilter(NoConnectionPoolFilter())


def set_default():
    _l = logging.getLogger(__name__.split(".")[0])

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    _l.addHandler(ch)

    return _l


levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]

logger = set_default()
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger.setLevel(logging.INFO)


def set_level(lvl):
    for h in logger.handlers:
        h.setLevel(lvl)
    logger.setLevel(lvl)


def set_format(f):
    _f = logging.Formatter(f)
    for h in logger.handlers:
        h.setFormatter(_f)


def add_file_handler(log_file):
    file_handler = logging.FileHandler(os.path.abspath(log_file), mode="w")
    logger.addHandler(file_handler)


class LogWriter:
    def __init__(self, log, level, stacklevel_offset=2) -> None:
        """Wrap logger in a write interface. Use this where a file descriptor is needed to
        be redirected to the log framework. Use stacklevel to indicate h

        Args:
            log (_type_): logger
            level (_type_): level of logger
            stacklevel_offset (int, optional): Number of frames to ignore in the call stack to find the
            correct model which should be displayed . Defaults to 2.
        """
        self.log = log
        self.level = level
        self.stacklevel_offset = stacklevel_offset

    def writelines(self, data: List[str]):
        for l in data:
            self.write(l)
            self.write(os.linesep)

    def write(self, data: str):
        for l in data.strip().splitlines():
            self.log.log(self.level, l.rstrip(), stacklevel=self.stacklevel_offset)

    def flush(self):
        pass

    @classmethod
    def info(cls, stacklevel=2):
        return cls(logger, logging.INFO, stacklevel_offset=stacklevel)

    @classmethod
    def info2(cls):
        """remove 2 levels from call stack"""
        return cls(logger, logging.INFO, stacklevel_offset=3)

    @classmethod
    def debug(cls, stacklevel=2):
        return cls(logger, logging.DEBUG, stacklevel_offset=stacklevel)


def timing(func):
    @wraps(func)
    def _timing(*args, **kwargs):
        ts = it.default_timer()
        logger.debug(f"{func.__name__}>")
        result = func(*args, **kwargs)
        logger.debug(
            f"{func.__name__}: took {it.default_timer() - ts:2,.2f} seconds",
            stacklevel=2,
        )
        return result

    return _timing


class TimeIt:
    def __init__(self) -> None:
        self.start = 0.0
        self.end = 0.0
        self.round = 0.0

    def __enter__(self):
        self.start = it.default_timer()
        self.round = self.start
        return self

    def __exit__(self, type, value, traceback):
        self.end = it.default_timer()

    @property
    def t(self):
        return self.end - self.start

    def str(self, format_str="2.4f"):
        return f"{format(self.t, format_str)} s"

    def round_str(self, format_str="2.4f"):
        now = it.default_timer()
        ret = f"{format(now - self.round, format_str)} s"
        self.round = now
        return ret
