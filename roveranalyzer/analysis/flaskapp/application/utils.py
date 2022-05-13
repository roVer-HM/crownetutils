import threading
from collections import defaultdict
from functools import _make_key, lru_cache, update_wrapper
from typing import Any


class threaded_lru:

    cache_store = {}

    def __init__(self, maxsize=128, typed=False):
        self.maxsize = maxsize
        self.typed = typed
        self.lock_dict = defaultdict(threading.Lock)

    def __call__(self, func):

        self.func = lru_cache(self.maxsize, self.typed)(func)

        def thread_lru(*args, **kwds):
            key = _make_key(args, kwds, typed=self.typed)
            with self.lock_dict[key]:
                return self.func(*args, **kwds)

        thread_lru.__setattr__("cache_info", self.cache_info)
        thread_lru.__setattr__("cache_clear", self.cache_clear)
        thread_lru.__setattr__("__name__", self.func.__name__)
        return thread_lru

    def cache_info(self):
        return self.func.cache_info()

    def cache_clear(self):
        self.func.cache_clear()

    @classmethod
    def log_cache_infos(cls):
        for k, v in cls.cache_store:
            pass
