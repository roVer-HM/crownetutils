from __future__ import annotations

import sys
from copy import deepcopy
from typing import List, Tuple

import pandas as pd

from crownetutils.analysis.hdf.provider import BaseHdfProvider


def save_box_plot_to_hdf(bbox: pd.Series, hdf: BaseHdfProvider, base_name: str = ""):
    box_group = "{}boxes".format(base_name)
    flier_group = "{}fliers".format(base_name)
    bbox = pd.DataFrame.from_records(data=bbox.values, index=bbox.index)
    fliers = (
        bbox["fliers"].explode().to_frame().set_axis(["fliers"], axis=1).astype(float)
    )
    bbox = bbox.drop(columns=["fliers"])
    hdf.write_frame(frame=bbox, group=box_group)
    hdf.write_frame(frame=fliers, group=flier_group)


class GroupWithAttributes:
    def __init__(self, group_name, columns: List[str] | None = None, **kwargs) -> None:
        self.group_name = group_name
        self.columns = columns
        self.attributes = kwargs
        self._group_missing = False
        self._missing_attributes = {}
        self._missing_columns = []

    def set_group_missing(self):
        self._group_missing = True

    def missing_attr(self, k, v=None):
        self._missing_attributes[k] = v

    def missing_column(self, c):
        self._missing_columns.append(c)

    def write_diff(self, writer=sys.stdout, prefix=""):
        if self._group_missing:
            writer.writelines([f"{prefix}{self.group_name} (MISSING!)"])
        elif len(self._missing_attributes) > 0 or len(self._missing_columns) > 0:
            writer.writelines([f"{prefix}{self.group_name}"])

        if len(self._missing_attributes):
            l = []
            for k, v in self._missing_attributes.items():
                if v is None:
                    l.append(f"{prefix}|--{k} (MISSING)")
                else:
                    l.append(
                        f"{prefix}|--{k}: {self.attributes[k]} != {v} (WRONG VALUE)"
                    )
            writer.writelines(l)

        # todo columns


class ExpectedHdfContent:
    """Collect expected content (i.e. groups and attributes)  of a HDF file and provide a test
    procedure to check if a provided HDF-File conforms to the content defined here.
    """

    def __init__(self, groups: List[GroupWithAttributes] | None = None) -> None:
        if groups is not None:
            self.groups: List[GroupWithAttributes] = groups
        else:
            self.groups: List[GroupWithAttributes] = []

    def add_group(self, g: GroupWithAttributes | str, **kwargs) -> ExpectedHdfContent:
        if isinstance(g, str):
            self.groups.append(GroupWithAttributes(g, **kwargs))
        else:
            for k, v in kwargs.items():
                g.attributes[k] = v
            self.groups.append(g)
        return self

    def add_groups(self, groups: List[str], **kwargs) -> ExpectedHdfContent:
        """Add multiple groups with SAME attribute set to ExpectedContent. Use None value to only check
        the presents of attributes and not the concrete value.

        Args:
            groups (List[str]): List of string group paths expected to be in HDF file

        Returns:
            ExpectedContent: _description_
        """
        for g in groups:
            self.add_group(g, **kwargs)

        return self

    def append_attributes_if_missing(self, **kwargs):
        for g in self.groups:
            for attr, value in kwargs.items():
                if attr not in g.attributes:
                    g.attributes[attr] = value

    def write_diff(self, writer=sys.stdout, header="Expected Content:"):
        writer.writelines([header])
        for g in self.groups:
            g.write_diff(writer=writer)

    def test_hdf(self, hdf: BaseHdfProvider) -> Tuple[bool, ExpectedHdfContent]:
        is_same = True
        diff = deepcopy(self)
        for g in diff.groups:
            if not hdf.contains_group(g.group_name):
                is_same = False
                g.set_group_missing()
            else:
                for attr, val in g.attributes.items():
                    if not hdf.has_attribute(attr_key=attr, group=g.group_name):
                        # todo mark difference
                        g.missing_attr(attr)

                        is_same = False
                    elif (
                        val is not None
                        and hdf.get_attribute(attr_key=attr, group=g.group_name) != val
                    ):
                        is_same = False
                        diff.missing_attr(attr, val)

                # todo columns

        return is_same, diff
