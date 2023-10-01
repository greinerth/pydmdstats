import pytest
import inspect
import os
import sys
import time
import numpy as np

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from scripts.util.stats import runtime_stats

def test_stats():
    STD_TIME = 1e-3
    N_RUNS = 100

    def dumm_func(sleep_time: float, std: float):
        tts = np.random.normal(0, std) + sleep_time
        time.sleep(tts)

    wrapped_func = runtime_stats(False)(dumm_func)
    for _ in range(N_RUNS):
        wrapped_func(10e-3, STD_TIME)

    mean = wrapped_func.mean
    std = np.sqrt(wrapped_func.var)

    assert mean == pytest.approx(10e-3)
    assert std == pytest.approx(STD_TIME)

