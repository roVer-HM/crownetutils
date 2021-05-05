import logging


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
