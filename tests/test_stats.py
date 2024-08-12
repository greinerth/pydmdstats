from __future__ import annotations

import time

import numpy as np
import pytest
from varprodmdstatspy.util.stats import Stats, runtime_stats


def test_stats() -> None:
    """Test running mean/variance."""
    stats = Stats()
    for _ in range(100):
        stats.push(5.0)

    assert stats.mean == 5.0
    assert stats.var == 0.0
    assert stats.std == 0.0

    stats.reset()

    msg = "Need more samples!"
    with pytest.raises(ZeroDivisionError, match=msg):
        _ = stats.var


def test_runtime_stats() -> None:
    """Test runtime statistics."""
    STD_TIME = 1e-3
    N_RUNS = 100
    generator = np.random.Generator(np.random.PCG64())

    def dumm_func(sleep_time: float, std: float):
        tts = generator.normal(0, std) + sleep_time
        time.sleep(tts)

    wrapped_func = runtime_stats(False)(dumm_func)
    for _ in range(N_RUNS):
        wrapped_func(10e-3, STD_TIME)

    mean = wrapped_func.mean
    std = np.sqrt(wrapped_func.var)

    assert mean == pytest.approx(10e-3, 0.1)
    assert std == pytest.approx(STD_TIME, 0.1)
