import pandas as pd
from geopandas import GeoDataFrame


class GeoProvider:
    """
    IHdfProvider wrapper which simplifies the direct return of GeoDataFrames
    using the 'get_dataframe()' method and the slicing api. The provider
    must provide the specifc to_geo(df: pd.DataFrame) -> GeoDataFrame method!

    Example:

    provider = SomeProvider(....)
    provider.geo.get_dataframe()  # get the full dataframe managed by 'provider' and
                                  # will transfrom the dataframe to a GeoDataFrame

    provider.geo[IndexSlice[...]]   # same as above but with on disk filtering before
                                    # transformation to GeoDataFrame takes place

    """

    def __init__(self, provider, to_crs) -> None:
        self._provider = provider
        self._to_crs = to_crs
        self._check()

    def _check(self):
        """
        ensure that provider has 'x' and 'y' values either in the index
        or as columns
        """
        in_key = all([i in self._provider.index_order().values() for i in ["x", "y"]])
        in_col = all([i in self._provider.columns() for i in ["x", "y"]])
        if not (in_key or in_col):
            raise ValueError(
                "provider does not have 'x' and 'y' in either index or columns"
            )

    @property
    def provider(self):
        return self._provider

    def _to_geo(self, df: pd.DataFrame) -> GeoDataFrame:
        return self._provider._to_geo(df, self._to_crs)

    def get_dataframe(self) -> GeoDataFrame:
        df = self._provider.get_dataframe()
        return self._to_geo(df)

    def __getitem__(self, value: any) -> GeoDataFrame:
        df = self._provider.__getitem__(value)
        return self._to_geo(df)
