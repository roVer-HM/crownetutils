from typing import Dict, List

from roveranalyzer.analysis.hdf.groups import HdfGroups
from roveranalyzer.analysis.hdf.provider import IHdfProvider


class GenericHdfProvider(IHdfProvider):
    """
    Generic provider to access any HDF store using select_where or dataframe accessors
    """

    def __init__(
        self, hdf_path: str, index_order: List[str], columns, default_index=None
    ):
        super().__init__(hdf_path)
        self._index_order: dict = {i: v for i, v in enumerate(index_order)}
        self._column_names: List[str] = columns
        self._default_index: str = (
            self._index_order[0] if default_index is None else default_index
        )

    def group_key(self) -> str:
        return HdfGroups.COUNT_MAP

    def index_order(self) -> Dict:
        return self._index_order

    def columns(self) -> List[str]:
        return self._column_names

    def default_index_key(self) -> str:
        return self._default_index
