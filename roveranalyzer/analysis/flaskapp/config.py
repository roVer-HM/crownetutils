from distutils.debug import DEBUG

CacheConfig = {
    "DEBUG": True,  # some Flask specific configs
    "CACHE_TYPE": "MemcachedCache",  # Flask-Caching related configs
    "CACHE_DEFAULT_TIMEOUT": 300,
}


class FlaskConfig:
    DEBUG = False
    TESTTING = False
    FOO = "Bar"


class FlaskConfigDbg(FlaskConfig):
    DEBUG = True
