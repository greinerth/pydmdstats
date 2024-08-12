"""Runtime statistics for execution times"""
from __future__ import annotations

import logging
import timeit
from typing import Any, Callable

import numpy as np

logging.basicConfig(level=logging.INFO, filename=__name__)


class Stats:
    """Calculate running mean and variance."""

    __slots__ = ["_mean", "_var_hat", "_cnt"]

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._var_hat: float = 0.0
        self._cnt: int = 1

    def push(self, val: float) -> None:
        """Add new value to calculate statistics

        :param val: New value for running mean/variance calculation.
        :type val: float
        """
        delta = val - self._mean
        self._mean += delta / float(self._cnt)
        self._var_hat += (val - self._mean) * delta
        self._cnt += 1

    def reset(self) -> None:
        """Reset all values."""
        self._mean = 0.0
        self._var_hat = 0.0
        self._cnt = 1

    @property
    def mean(self) -> float:
        """Get calculated mean.

        :return: mean value.
        :rtype: float
        """
        return self._mean

    @property
    def var(self) -> float:
        """Get sample variance.

        :raises ZeroDivisionError: If no values were added for variance calculation.
        :return: Sample variance
        :rtype: float
        """
        if self._cnt == 1:
            msg = "Need more samples!"
            raise ZeroDivisionError(msg)
        return self._var_hat / float(self._cnt - 1)

    @property
    def std(self) -> float:
        """Get sample standard deviation.

        :return: Sample standard deviation.
        :rtype: float
        """
        return float(np.sqrt(self.var))


class ExecutionStats(Stats):
    """Measure the execution time of a function/method"""

    __slots__ = ["_func", "_verbose", "_min", "_max"]

    def __init__(self, function: Callable[[Any], Any], verbose: bool = True) -> None:
        super().__init__()
        self._func = function
        self._verbose: bool = verbose
        self._min: float = float("inf")
        self._max: float = 0.0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        t_1 = timeit.default_timer()
        res = self._func(*args, **kwds)
        delta_t = timeit.default_timer() - t_1
        self.push(delta_t)

        self._min = min(delta_t, self._min)
        self._max = max(delta_t, self._max)

        return res

    @property
    def min(self) -> float:
        """Return minimum encountered execution time

        :return: Minimum encountered execution time_description_
        :rtype: float
        """
        return self._min

    @property
    def max(self) -> float:
        """Return maximum encountered execution time

        :return: Maximum encountered execution time_description_
        :rtype: float
        """
        return self._max

    def reset(self) -> None:
        """Reset the stats"""
        super().reset()
        self._min = float("inf")
        self._max = 0.0

    def __del__(self) -> None:
        if self._verbose:
            stats = f"\n{self._func.__name__} stats:\n"
            stats += f"Mean execution time: {self.mean} [s]\n"
            if self._cnt > 1:
                stats += f"Var execution time: {self.var} [s]\n"
            logging.info(stats)


def runtime_stats(verbose: bool = True) -> Callable[[Any], Any]:
    """Runtime stats wrapper

    :param verbose: Set verbosity level, defaults to True
    :type verbose: bool, optional
    :return: ExecutionStats object
    :rtype: Callable[[Any], Any]
    """

    def decorator(func: Callable[[Any], Any]) -> ExecutionStats:
        return ExecutionStats(func, verbose)

    return decorator
