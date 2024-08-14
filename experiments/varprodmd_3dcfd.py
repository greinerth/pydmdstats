from __future__ import annotations

import timeit
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
from pydmd import BOPDMD, VarProDMD


def test_3dcfd(
    path2data: str, split: float = 0.7
) -> tuple[list[dict[str, Any]], np.ndarray, int]:
    """Test 3D CFD example

       Split data into training set and test model on unseen timesteps (extrapolation).
       In addition record the time required for optimization.

    :param path2data: Path to datasetr
    :type path2data: str
    :param split: Splitting parameter to split data to training- and test set, defaults to 0.7
    :type split: float, optional
    :raises ValueError: If parameter `split` not in (0, 1)
    :raises FileNotFoundError: If dataset cannot be found
    :return: List of results of each trial, time and number of training samples
    :rtype: tuple[list[dict[str, Any]], np.ndarray, int]
    """
    if not 0.0 < split < 1.0:
        msg = "'split' must lie in (0, 1)!"
        raise ValueError(msg)

    if not Path(path2data).exists():
        msg = f"{path2data} does not exist!"
        raise FileNotFoundError(msg)

    h5file = h5.File(str(path2data))
    data = h5file["density"]
    msg = f"Data shape : {data.shape}"
    n_train = (1.0 - split) * data.shape[1]

    time = np.array(h5file["t-coordinate"][:-1])

    results: list[dict[str, Any]] = []

    for trial in range(data.shape[0]):
        current_data = data[trial]
        dataflat = np.zeros((np.prod(current_data.shape[1:]), current_data.shape[0]))

        for i in range(current_data.shape[0]):
            dataflat[:, i] = np.ravel(data[i, ...], "F")

        vardmd = VarProDMD()
        bopdmd = BOPDMD(trial_size=n_train)

        t0 = timeit.default_timer()
        bopdmd.fit(current_data[:, :n_train], time[:n_train])
        dtbop = timeit.default_timer() - t0

        t0 = timeit.default_timer()
        vardmd.fit(current_data[:, :n_train], time[:n_train])
        dtvar = timeit.default_timer() - t0

        bopmrse = np.linalg.norm(current_data - bopdmd.forecast(time), axis=0)
        varmrse = np.linalg.norm(current_data - vardmd.forecast(time), axis=0)

        results.append({"method": "BOPDMD", "run": trial, "dt": dtbop, "mrse": bopmrse})

        results.append(
            {"method": "VarProDMD", "run": trial, "dt": dtvar, "mrse": varmrse}
        )

    return results, time, n_train
