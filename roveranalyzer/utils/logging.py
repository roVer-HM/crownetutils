import logging
import timeit as it
from functools import wraps


class NoConnectionPoolFilter(logging.Filter):
    def filter(self, record):
        return not (record.module == "connectionpool" and record.levelname == "DEBUG")


# logging.root.addFilter(NoConnectionPoolFilter())


def set_default():
    _l = logging.getLogger("roveranalyzer")

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


def set_level(lvl):
    logger.setLevel(lvl)


def set_format(f):
    _f = logging.Formatter(f)
    logger.handlers[0].setFormatter(_f)


levels = [logging.ERROR, logging.WARN, logging.INFO, logging.DEBUG]

logger = set_default()
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logger.setLevel(logging.INFO)


def timing(func):
    @wraps(func)
    def _timing(*args, **kwargs):
        ts = it.default_timer()
        logger.debug(f"{func.__name__}>")
        result = func(*args, **kwargs)
        logger.info(f"{func.__name__}: took {it.default_timer() - ts:2.4f} seconds")
        return result

    return _timing
