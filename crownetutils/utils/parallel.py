""" Multiprocessing map function helper in both args/kwargs versions.

"""
from __future__ import annotations

import traceback
from functools import partial
from itertools import repeat
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from typing import Any, List, Literal, Tuple, Type

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


PoolKind = Literal["process", "thread"]


def get_pool(size, pool_type, kind):
    if kind == "process":
        return get_context(pool_type).Pool(processes=size)
    elif kind == "thread":
        return ThreadPool(processes=size)


def run_kwargs_map(
    func,
    kwargs_iter,
    pool_size: int = 10,
    pool_type: str = "spawn",
    raise_on_error: bool = True,
    append_args: bool = False,
    filter_id: int | List[int] | None = None,
    pool_kind: PoolKind = "process",
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
        with get_pool(size=pool_size, pool_type=pool_type, kind=pool_kind) as pool:
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


def run_item(item: ExecutionItem):
    return item()


def run_items(
    items: List[ExecutionItem],
    pool_size: int = 10,
    pool_type: str = "spawn",
    raise_on_error: bool = True,
    unpack: bool = True,
    filter_id: int | List[int] | None = None,
    pool_kind: PoolKind = "process",
):
    map: List[ResultItem] = []
    if filter_id is not None:
        filter_id = [filter_id] if isinstance(filter_id, int) else filter_id
        items = [items[i] for i in filter_id]

    with get_pool(size=pool_size, pool_type=pool_type, kind=pool_kind) as pool:
        map = pool.map(run_item, iterable=items)

    if raise_on_error:
        ret_data = []
        ret_err: List[ResultItem] = []
        for ret in map:
            if ret.ok:
                if unpack:
                    ret_data.append(ret.value)
                else:
                    ret_data.append(ret)
            else:
                ret_err.append(ret)
        if len(ret_err) > 0:
            for ret in ret_err:
                logger.error(ret.exception)
            raise ValueError(f"{len(ret_err)} out of {len(items)} failed")
        return ret_data
    else:
        if unpack:
            return [i.value for i in map]
        else:
            return map


def run_args_map(
    func,
    args_iter,
    pool_size: int = 10,
    pool_type: str = "spawn",
    raise_on_error: bool = True,
    append_args: bool = False,
    filter_id: int | List[int] | None = None,
    pool_kind: PoolKind = "process",
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
        with get_pool(size=pool_size, pool_type=pool_type, kind=pool_kind) as pool:
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


class ExecutionItem:
    def __init__(self, fn, args=None, kwargs=None) -> None:
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.postprocessing = []

    def add_post_function(self, fn, *args, **kwargs):
        self.postprocessing.append((fn, args, kwargs))

    def __call__(self) -> Any:
        try:
            _args = [] if self.args is None else self.args
            _kwargs = {} if self.kwargs is None else self.kwargs
            ret = self.fn(*_args, **_kwargs)
        except Exception as e:
            traceback.print_exc()
            ret = ResultItem(None, self, e, self.fn.__name__)
        else:
            try:
                for _pfn, _pargs, _pkwargs in self.postprocessing:
                    ret = _pfn(ret, *_pargs, **_pkwargs)
            except Exception as e:
                traceback.print_exc()
                ret = ResultItem(None, self, e, _pfn.__name__)
            else:
                ret = ResultItem(ret, self, None)

        return ret


class ResultItem:
    def __init__(
        self, ret=None, exec_item: ExecutionItem = None, exception=None, fn_name=None
    ) -> None:
        self.ret = ret
        self.exec_item = exec_item
        self.exception = exception
        self.exception_func = fn_name

    @property
    def ok(self):
        return self.exception is None

    @property
    def value(self) -> Any:
        return self.ret

    def __repr__(self) -> str:
        e = (
            "Ok"
            if self.ok
            else f"Err({self.exception.__class__.__name__}-'{self.exception}')"
        )
        return f"<{self.__class__.__module__}.{self.__class__.__name__}[{e}] at {hex(id(self))}"
