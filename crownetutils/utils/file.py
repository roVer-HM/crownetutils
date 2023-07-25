import contextlib
import gzip
import json
import os

from crownetutils.utils.yesno import query_yes_no


@contextlib.contextmanager
def open_txt_gz(path):
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


def ask_rm(path, default=False):
    if os.path.exists(path):
        if query_yes_no(f"Delete File: {path}:", default):
            os.remove(path)


def read_lines(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.readlines()
    else:
        raise FileNotFoundError("File not found {}".format(path))


def read_json_to_dict(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    else:
        raise FileNotFoundError("File not found {}".format(path))


def clean_dir_name(dir_name):
    ret = dir_name.replace(".", "_")
    ret = ret.replace("-", "_")
    return ret
