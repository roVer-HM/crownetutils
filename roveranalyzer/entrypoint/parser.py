import argparse
import copy
import re
from typing import Sequence, Text

from roveranalyzer.dockerrunner.dockerrunner import DockerCleanup, DockerReuse


def filter_options(_args, prefix):
    res = []
    res.extend([i for i in _args if i.startswith(prefix)])
    return res


class SubstituteAction(argparse.Action):
    def __init__(self, option_strings: Sequence[Text], dest: Text, **kwargs):
        self.sub_f = kwargs.pop("sub_action")
        self.do_on = kwargs.pop("do_on")
        _default = kwargs.pop("default")
        _help = f"{kwargs.pop('help')} (Default: '{_default}')"
        if _default in self.do_on:
            _default = self.sub_f(_default)
        super().__init__(option_strings, dest, default=_default, help=_help, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        if values in self.do_on:
            setattr(namespace, self.dest, self.sub_f(values))
        else:
            setattr(namespace, self.dest, values)


class SimulationArgAction(argparse.Action):
    def __init__(self, option_strings: Sequence[Text], dest: Text, *args, **kwargs):
        self.dest = dest
        self.prefix = kwargs.pop("prefix")
        help_text = kwargs.pop("help")
        help_default = " ".join(kwargs["default"].to_list(self.prefix))
        help_text = f"{help_text} [{help_default}]"
        super().__init__(
            option_strings, dest, nargs="*", required=False, help=help_text, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None, *args, **kwargs):
        if option_string is None:
            return
        if not option_string.startswith(self.prefix):
            raise ValueError(
                f"Action only expected options staring with `{self.prefix}` got `{option_string}`"
            )
        sanitized_opt = option_string[len(self.prefix) :]
        if len(sanitized_opt) == 0:
            raise ValueError(f"do not use {self.prefix} alone.")
        if not sanitized_opt.startswith("-"):
            sanitized_opt = f"--{sanitized_opt}"

        ns_dest = getattr(namespace, self.dest)
        ns_dest.add_override(sanitized_opt, values)
        setattr(namespace, self.dest, ns_dest)


def test_parser(args=None, parse_args=False):
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", dest="data_dir", help="Base data directory.")

    if parse_args:
        return parser.parse_known_args(args=args)
    else:
        return parser


class ArgList:
    def __init__(self):
        self.data = []

    @classmethod
    def from_flat_list(cls, data):
        """
        Expects a list of the for [ A, A, A, A, ...]
        * A must be one of the following [ str, int, float]
        * A will be cast to string
        * This will try the find the key-value pairs where keys start with [-|--]
        * leading runtime and executbales will be found if they do not start with [-|--] which they normally should not.
        """
        first_option_found = False  # any entry starting with at least one '-'
        arg_ = []
        current_key = None
        values = None
        for item in data:
            # if the first option item is found deactivate first_option_found
            first_option_found = first_option_found or item.startswith("-")
            if item.startswith("-") and current_key is not None:
                # write previous pair
                arg_.append(cls.build_arg(current_key, values))
                # set new key with empty value
                current_key = item
                values = None
            elif item.startswith("-") and current_key is None:
                # new key, no previous pair present
                current_key = item
                values = None
            elif not first_option_found:
                # there war no option jet This is probability the first item (aka. command name)
                arg_.append(cls.build_arg(item, None))
            else:
                # item is value
                if values is None:
                    values = [item]
                else:
                    values.append(item)
        if current_key is not None:
            arg_.append(cls.build_arg(current_key, values))
        return cls.from_list(arg_)

    @classmethod
    def from_list(cls, data):
        """
        Expects a list of the form [ [A,B], [A,B], [A,B], [A,B], [A,B], ... ]
        A must be a string
        B must be one of the following [ None, str, int, float, or list of the previously mentioned]
        B will be cast to string (or list of strings)
        """
        if any([len(_arg) != 2 for _arg in data]):
            raise ValueError("Expected list of 2-element lists")
        _data = copy.deepcopy(data)
        _data = [cls.build_arg(a, b) for a, b in _data]
        _obj = cls()
        _obj.data = _data
        return _obj

    @classmethod
    def build_arg(cls, key, value):
        return [cls.check_key(key), cls.check_val(value)]

    @staticmethod
    def check_key(key):
        return str(key)

    @staticmethod
    def check_val(value):
        """
        Only string or list of strings allowed. Change type wherever possible
        """
        # treat empty lists as None
        if value is None or (type(value) == list and len(value) == 0):
            return None
        # cast numbers to strings
        if type(value) in [int, float]:
            value = str(value)
        # cast list of numbers to list of string
        if type(value) == list and any([type(i) in [int, float] for i in value]):
            value = [str(i) for i in value]
        # unpack list of len == 1
        if type(value) == list and len(value) == 1:
            value = value[0]
        # only accept strings and list (everything that is not a string by now is an error)
        if type(value) not in [str, list]:
            raise ValueError(f"expected str or list got {type(value)}")
        return value

    def remove_key(self, key):
        self.data = [_arg for _arg in self.data if _arg[0] != key]

    def contains_key(self, key):
        return any([_arg[0] == key for _arg in self.data])

    def get_value(self, key, default=None):
        if not self.contains_key(key):
            return default
        return self.data[self.key_index(key)[0]][1]

    def key_index(self, key):
        return [idx for idx, _arg in enumerate(self.data) if _arg[0] == key]

    def add(self, key, value=None, pos=-1):
        """
        add key. Raise error if key already exists. If pos is given add arg at this
        index. If the index is bigger than the length just append at the end.
        """
        if self.contains_key(key):
            raise ValueError(f"key '{key}' already in ArgList")
        if pos < 0 or pos >= len(self.data):
            self.data.append(self.build_arg(key, value))
        elif pos >= 0 and pos < len(self.data):
            self.data.insert(pos, self.build_arg(key, value))

    def add_override(self, key, value=None):
        """
        add key, override any (one or more) existing keys
        """
        self.remove_key(key)
        self.data.append(self.build_arg(key, value))

    def add_if_missing(self, key, value=None, pos=-1):
        """
        check if key exists, if not add item. This does not compares the value
        """
        if not self.contains_key(key):
            self.data.append(self.build_arg(key, value))

    def append(self, key, value=None):
        """
        append key without checking if it already exists. ArgList with multiple keys are valid.
        """
        self.data.append(self.build_arg(key, value))

    def update_value(self, key, value):
        """
        find key and update value. If key is missing add it. Raise error if multiple keys are found
        """
        idx = self.key_index(key)
        if len(idx) > 1:
            raise ValueError(f"expected only one occurrence of key {key}")
        elif len(idx) == 1:
            _arg = self.data[idx[0]]
            self.data[idx[0]] = self.build_arg(key, value)
        else:
            # key does not exist
            self.data.append([key, self.check_val(value)])

    def update_index(self, index, key, value=None):
        """
        replace key/value at given index
        """
        self.data[index] = self.build_arg(key, value)

    def merge(self, other, how="add_override"):
        """
        merge other ArgList into this one. Choose function to use merge
        """
        m_ = getattr(self, how, None)
        if m_ is None:
            raise RuntimeError(
                f"method {how} not found. Choose one of the add/append methods to use"
            )
        for _args in other.data:
            m_(_args[0], _args[1])

    def raw(self):
        return list(self.data)

    def to_list(self, prefix_key=""):
        arg_list = []
        for key, value in self.data:
            arg_list.append(f"{prefix_key}{key}")
            if value is None:
                continue
            if type(value) == list:
                arg_list.extend(value)
            else:
                arg_list.append(value)
        return arg_list

    def to_string(self, sep=" "):
        return sep.join(self.to_list())
