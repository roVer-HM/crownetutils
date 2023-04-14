""" Multiprocessing map function helper in both args/kwargs versions.

"""
from __future__ import annotations

import traceback
from functools import partial
from itertools import repeat
from multiprocessing import get_context
from typing import Any, List, Tuple

from crownetutils.utils.logging import logger


def kwargs_with_try(
    func, kwargs: dict, append_args: bool = False
) -> Tuple[bool, Any] | Tuple[bool, Tuple[dict, Any]]:
    """Execute func in try-except block and return result without rasing any exception.

    Args:
        func (callable): Function to execute.
        kwargs (dict): Dictionary arguments passed to func.
        append_args (bool, optional): Append argument dictionary in return type. Defaults to False.

    Returns:
        Tuple[bool, Any]|Tuple[bool, Tuple[dict, Any]]: Result code, result and used arguments if 'append_args' is true
    """
    try:
        logger.debug(f"Execute {kwargs}")
        ret = func(**kwargs)
        if append_args:
            return (True, (ret, kwargs))
        else:
            return (True, ret)
    except Exception as e:
        logger.exception(f"Error in run for args {kwargs} message: {e}")
        trace = traceback.format_exc()
        return (False, f"Error in args: {kwargs} message: {e}\n{trace}")


def run_kwargs_map(
    func,
    kwargs_iter,
    pool_size: int = 10,
    pool_type: str = "spawn",
    raise_on_error: bool = True,
    append_args: bool = False,
    filter_id: int | List[int] | None = None,
) -> List[Tuple[bool, Any]] | List[Any]:
    """Execute `func` in parallel

    Args:
        func (callable): function to be executed
        kwargs_iter (int): used keyword arguments
        pool_type: (str): Defaults to spawn
        pool_size (int, optional): Number of processes. Defaults to 10.
        raise_on_error: (bool): Will raise Error after checking result. Defaults to True
        append_args: (bool): If True append kwargs to return value
        filter_id (int | List[int] | None, optional): Only run selection of runs given by kwargs_iter. Defaults to None.

    Returns:
        List[Tuple[bool, Any]] | List[Any]: Either list of result codes and result or results only if raise_on_err is True.

    """
    map = [(False, "No results")]
    if filter_id is not None:
        filter_id = [filter_id] if isinstance(filter_id, int) else filter_id
        kwargs_iter = [kwargs_iter[i] for i in list(filter_id)]

    if len(kwargs_iter) == 1:
        map = [kwargs_with_try(func, kwargs_iter[0], append_args=append_args)]
    else:
        with get_context(pool_type).Pool(processes=pool_size) as pool:
            map = pool.starmap(
                partial(kwargs_with_try, append_args=append_args),
                zip(repeat(func), kwargs_iter),
            )

    if raise_on_error:
        ret_data = []
        ret_err = []
        for ret, data in map:
            if ret:
                ret_data.append(data)
            else:
                ret_err.append(data)
        if len(ret_err) > 0:
            for ret in ret_err:
                logger.error(ret)
            raise ValueError(f"{len(ret_err)} out of {len(kwargs_iter)} failed")
        return ret_data
    else:
        return map


def args_with_try(
    func, args: List[Any], append_args: bool = False
) -> Tuple[bool, Any] | Tuple[bool, Tuple[List[Any], Any]]:
    """Execute func in try-except block and return result without rasing any exception.

    Args:
        func (callable): Function to execute.
        args (List[Any]): Positional arguments passed to func.
        append_args (bool, optional): Append used arguments to return value. Defaults to False.

    Returns:
        Tuple[bool, Any]|Tuple[bool, Tuple[List[Any], Any]]: Return code and result and argument list if append_args is true.
    """
    try:
        logger.debug(f"Execute {args}")
        ret = func(*args)
        if append_args:
            return (True, (ret, args))
        else:
            return (True, ret)

    except Exception as e:
        logger.info(f"Error in run for args {args} message: {e}")
        return (False, f"Error in args: {args} message: {e}")


def run_args_map(
    func,
    args_iter,
    pool_size: int = 10,
    pool_type: str = "spawn",
    raise_on_error: bool = True,
    append_args: bool = False,
    filter_id: int | List[int] | None = None,
) -> List[Tuple(bool, Any)] | List[Any]:
    """Execute `func` in parallel with the possibility to debug single runs if
    necessary. To do this add the arg_iter index of the run(s) to debug in the
    filter_id function argument.

    Args:
        func (callable): function to be executed
        args_iter (int): used arguments
        pool_type: (str): Defaults to spawn
        pool_size (int, optional): Number of processes. Defaults to 10.
        raise_on_error: (bool): Will raise Error after checking result. Defaults to True
        append_args: (bool): If True append arguments to return value
        filter_id (int | List[int] | None, optional): Only run selection of runs given by kwargs_iter. Defaults to None.

    Returns:
        List[Tuple(bool, Any)] | List[Any]: Either list of result codes and result or results only if raise_on_err is True.

    """
    map = [(False, "No results")]
    if filter_id is not None:
        filter_id = [filter_id] if isinstance(filter_id, int) else filter_id
        args_iter = [args_iter[i] for i in filter_id]

    if len(args_iter) == 1:
        map = [args_with_try(func, args_iter[0], append_args=append_args)]
    else:
        with get_context(pool_type).Pool(processes=pool_size) as pool:
            map = pool.starmap(
                partial(args_with_try, append_args=append_args),
                zip(repeat(func), args_iter),
            )

    if raise_on_error:
        ret_data = []
        ret_err = []
        for ret, data in map:
            if ret:
                ret_data.append(data)
            else:
                ret_err.append(data)
        if len(ret_err) > 0:
            for ret in ret_err:
                logger.error(ret)
            raise ValueError(f"{len(ret_err)} out of {len(args_iter)} failed")
        return ret_data
    else:
        return map
