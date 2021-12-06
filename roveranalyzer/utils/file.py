import json
import os

from roveranalyzer.utils.yesno import query_yes_no


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
