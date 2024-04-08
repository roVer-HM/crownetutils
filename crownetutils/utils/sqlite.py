from __future__ import annotations

import timeit as it
from sqlite3 import Connection
from typing import ContextManager, Generator, List, Protocol, Tuple

import pandas as pd

from crownetutils.utils.logging import logger


class ConnectionProvider(Protocol):
    def con() -> ContextManager[Connection]:
        ...


class SortedChunkStream:
    """Stream sqlite table based on sorted id_column. The table must provide an INT uid column.

    Each generated chunk ensures that it contains all rows for values chunk['id_column'].unique(). For example if
    id_column is 'simtime' and t=5.0 is part of the chunk, the chunk contains all rows with t=5.0.
    """

    def __init__(
        self,
        sql_template: str,
        chunk_size: int,
        id_column: str = "simtime",
        initial_offset: int = 0,
    ) -> None:
        """

        Args:
            sql_template (str): Template string containing `lower_bound` and `upper_bound` template strings for uid Integer.
            chunk_size (int): Number of rows to query
            id_column (str, optional): The sorted column by which a `chunk` is defined. Defaults to "simtime".
            initial_offset (int, optional): Start point as row_id. Defaults to 0.
        """
        self.sql_template: str = sql_template
        self.chunk_size: int = chunk_size
        self.id_column: str = id_column
        self.initial_offset: int = initial_offset

    def split_chunk(
        self, data: pd.DataFrame, time_first: float, time_last: float
    ) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
        """Split of open chunk  before retuning all closed chunks."""
        if time_first != time_last:
            # more than one time!
            last_time_mask = data[self.id_column].iloc[-1] == data[self.id_column]
            rest_chunk = data[last_time_mask].copy()
            data = data[~last_time_mask].copy()
        else:
            rest_chunk = None

        return data, rest_chunk

    def chunk_stream(
        self, connection_provider: ConnectionProvider, with_con: bool = False
    ) -> Generator[pd.DataFrame]:
        chunks: List[pd.DataFrame] = []
        time_first: float = -1.0
        time_last: float = -1.0
        load_next: bool = True
        lower_bound: int = self.initial_offset
        upper_bound: int = lower_bound + self.chunk_size

        with connection_provider.con() as _con:
            num_chunks = 0
            while load_next:
                # todo pragma
                ts = it.default_timer()
                num_chunks += 1
                sql = self.sql_template.format(
                    lower_bound=lower_bound, upper_bound=upper_bound
                )
                data = pd.read_sql_query(sql, _con)
                logger.info(
                    f"{num_chunks}/? query uid(ROW_ID interval) {lower_bound:,d} < ROWID <= {upper_bound:,d}  chunk_size={upper_bound - lower_bound:,d} took {it.default_timer() - ts:2,.2f}s"
                )
                lower_bound = upper_bound
                upper_bound = lower_bound + self.chunk_size

                if data.empty:
                    # reached end of table stop processing
                    logger.info(f"reached end of table stop after {num_chunks}")
                    load_next = False
                    if len(chunks) == 0:
                        # in case  no chunks are left do not process further
                        break
                else:
                    if len(chunks) == 0:
                        # only set time_first if no older chunk is present.
                        time_first = data[self.id_column].iloc[0]
                    time_last = data[self.id_column].iloc[-1]
                    chunks.append(data)

                if time_first != time_last or load_next is False:
                    logger.info(f"{num_chunks}/? Process loaded chunk(s).")
                    # found more than one time process time
                    # keep last time out of processing as it might not be
                    if len(chunks) == 1:
                        data = chunks[0]
                    else:
                        data = pd.concat(chunks, axis=0)

                    # process chunk and return last time if present
                    data, rest_chunk = self.split_chunk(data, time_first, time_last)

                    if with_con:
                        yield data, _con
                    else:
                        yield data

                    chunks.clear()
                    time_first = -1
                    time_last = -1

                    if rest_chunk is not None:
                        time_first = rest_chunk[self.id_column].iloc[0]
                        time_last = rest_chunk[self.id_column].iloc[-1]
                        if time_first != time_last:
                            raise ValueError(
                                f"Rest chunk must be one time only!. Got {time_first} and {time_last}"
                            )
                        logger.info(
                            f"Add rest chunk back to chunk with {rest_chunk.shape[0]:,d} rows back to list of chunks. New start time time is {time_first}"
                        )
                        chunks.append(rest_chunk)
                else:
                    # Chunk does not contain all data for one time.
                    # Keep pulling data.
                    logger.info(
                        f"{num_chunks}/? loaded chunk(s) contains only one time value. Pull next chunk to check if there are more lines for that time value."
                    )
                    pass
