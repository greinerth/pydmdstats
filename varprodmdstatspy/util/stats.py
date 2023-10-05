"""Runtime statistics for execution times
"""
# import timeit
import timeit
from typing import Any, Callable


class ExecutionStats:
    """Measure the execution time of a function/method

    Raises:
        ZeroDivisionError: If only 1 samples was taken for
                           variance calulation

    """
    __slots__ = ['__func',
                 '__mean_time',
                 '__var_time_hat',
                 '__counter',
                 '__verbose',
                 '__min',
                 '__max']

    def __init__(self, function: Callable, verbose: bool = True) -> None:
        self.__func = function
        self.__mean_time: float = 0.
        self.__var_time_hat: float = 0.
        self.__counter: int = 1
        self.__verbose: bool = verbose
        self.__min: float = float('inf')
        self.__max: float = 0.

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        __prev_mean_time = self.__mean_time
        t_1 = timeit.default_timer()
        res = self.__func(*args, **kwds)
        delta_t = timeit.default_timer() - t_1
        delta = delta_t - __prev_mean_time
        self.__mean_time = __prev_mean_time + \
            delta / float(self.__counter)
        self.__var_time_hat += (delta_t - self.__mean_time) * delta
        self.__counter += 1

        if delta_t < self.__min:
            self.__min = delta_t

        if delta_t > self.__max:
            self.__max = delta_t

        return res

    @property
    def mean(self) -> float:
        """Get the mean execution time

        Returns:
            float: Execution time of several runs.
        """
        return self.__mean_time

    @property
    def var(self) -> float:
        """Get the variance of the execution time.

        Raises:
            ZeroDivisionError: If only one measurement was taken.

        Returns:
            float: Calculated variance.
        """
        if self.__counter == 1:
            raise ZeroDivisionError("Need more samples!")
        return self.__var_time_hat / float(self.__counter - 1)

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
        """ reset the stats
        """
        self.__mean_time = 0.
        self.__var_time_hat = 0.
        self.__min = float('inf')
        self.__max = 0.
        self.__counter = 1

    def __del__(self):
        if self.__verbose:
            __stats = f"\n{self.__func.__name__ } stats:\n"
            __stats += f"Mean execution time: {self.mean} [s]\n"
            if self.__counter > 1:
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
