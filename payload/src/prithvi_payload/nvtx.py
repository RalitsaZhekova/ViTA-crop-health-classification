"""Low-overhead NVTX ranges for Nsight Systems payload traces."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

_P = ParamSpec("_P")
_R = TypeVar("_R")


def enabled() -> bool:
    """Return whether optional payload NVTX annotations are enabled."""

    return os.environ.get("VITA_NVTX_ENABLED", "0").strip().casefold() in {
        "1",
        "true",
        "yes",
        "on",
    }


@contextmanager
def range(name: str) -> Iterator[None]:
    """Create an NVTX push/pop range without affecting ordinary execution."""

    pushed = False
    if enabled():
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.nvtx.range_push(name)
                pushed = True
        except (AttributeError, RuntimeError):
            pushed = False
    try:
        yield
    finally:
        if pushed:
            try:
                import torch

                torch.cuda.nvtx.range_pop()
            except (AttributeError, RuntimeError):
                pass


def annotate(name: str) -> Callable[[Callable[_P, _R]], Callable[_P, _R]]:
    """Decorate a function with an optional NVTX range."""

    def decorator(function: Callable[_P, _R]) -> Callable[_P, _R]:
        @wraps(function)
        def wrapped(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            with range(name):
                return function(*args, **kwargs)

        return wrapped

    return decorator
