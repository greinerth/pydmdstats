"""Runtime statistics for execution times"""

# import timeit
import timeit
from typing import Any, Callable

import numpy as np


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
        self._mean: float = 0.0
        self._var_hat: float = 0.0
        self._cnt: int = 1

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
            raise ZeroDivisionError("Need more samples!")
        return self._var_hat / float(self._cnt - 1)

    @property
    def std(self) -> float:
        """Get sample standard deviation.

        :return: Sample standard deviation.
        :rtype: float
        """
        return np.sqrt(self.var)


class ExecutionStats(Stats):
    """Measure the execution time of a function/method

    Raises:
        ZeroDivisionError: If only 1 samples was taken for
                           variance calulation

    """

    __slots__ = ["_func", "_verbose", "_min", "_max"]

    def __init__(self, function: Callable, verbose: bool = True) -> None:
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

        if delta_t < self._min:
            self._min = delta_t

        if delta_t > self._max:
            self._max = delta_t

        return res

    @property
    def min(self) -> float:
        """Return minimal encountered execution time

        Returns:
            float: minimal encountered execution time
        """
        return self.__min

    @property
    def max(self) -> float:
        """Return maximum encountered execution time

        Returns:
            float: maximum encountered execution time
        """
        return self.__max

    def reset(self):
        """reset the stats"""
        super().reset()
        self.__min = float("inf")
        self.__max = 0.0

    def __del__(self):
        if self._verbose:
            __stats = f"\n{self._func.__name__ } stats:\n"
            __stats += f"Mean execution time: {self.mean} [s]\n"
            if self._cnt > 1:
                __stats += f"Var execution time: {self.var} [s]\n"
            print(__stats)


def runtime_stats(verbose: bool = True) -> Callable:  # pylint: disable=unused-variable
    """runtime stats wrapper

    Args:
        func (Callable): Function to wrap

    Returns:
        ExecutionStats: ExecutionStats object
    """

    def decorator(func: Callable):
        return ExecutionStats(func, verbose)

    return decorator
