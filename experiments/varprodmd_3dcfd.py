"""
Conduct 3d fluid dynamic example taken from pdebench data
"""
from __future__ import annotations

import inspect

# import logging
import timeit
from pathlib import Path
from typing import Any, Generator

import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.patches import Rectangle
from pydmd import BOPDMD, VarProDMD
from tqdm import tqdm

# logging.basicConfig(level=logging.INFO, filename=__name__)
# logging.root.setLevel(logging.INFO)


def test_3dcfd(
    data: np.ndarray, time: np.ndarray, split: float = 0.3
) -> Generator[dict[str, Any], dict[str, Any]]:
    """Test 3D CFD example

       Split data into training set and test model on unseen timesteps (extrapolation).
       In addition record the time required for optimization.

    :param path2data: Path to dataset
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

    n_train = int((1.0 - split) * data.shape[1])

    for trial in range(data.shape[0]):
        current_data = data[trial]
        dataflat = np.zeros((np.prod(current_data.shape[1:]), current_data.shape[0]))

        for i in range(current_data.shape[0]):
            dataflat[:, i] = np.ravel(current_data[i, ...], "F")

        vardmd = VarProDMD(
            optargs={
                "method": "trf",
                "tr_solver": "exact",
                "loss": "linear",
                # "x_scale": "jac",
            },
            # compression=0.4
        )
        bopdmd = BOPDMD(trial_size=dataflat.shape[-1])

        t0 = timeit.default_timer()
        bopdmd.fit(dataflat[:, :n_train], time[:n_train])
        dtbop = timeit.default_timer() - t0

        t0 = timeit.default_timer()
        vardmd.fit(dataflat[:, :n_train], time[:n_train])
        dtvar = timeit.default_timer() - t0

        bopmrse = np.linalg.norm(dataflat - bopdmd.forecast(time), axis=0) / np.sqrt(
            dataflat.shape[0]
        )
        varmrse = np.linalg.norm(dataflat - vardmd.forecast(time), axis=0) / np.sqrt(
            dataflat.shape[0]
        )

        yield (
            {
                "method": "BOPDMD",
                "run": trial,
                "dt": dtbop,
                "mrse": bopmrse,
                "eigs": bopdmd.eigs,
            },
            {
                "method": "VarPorDMD",
                "run": trial,
                "dt": dtvar,
                "mrse": varmrse,
                "eigs": vardmd.eigs,
            },
        )


if __name__ == "__main__":
    plt.style.use("science")
    split = 0.3
    currentdir = Path(inspect.getfile(inspect.currentframe())).resolve().parent
    file = (
        currentdir
        / "data"
        / "3D"
        / "Train"
        / "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
    )
    file = str(file)

    if not Path(file).exists():
        msg = f"{file} does not exist!"
        raise FileNotFoundError(msg)

    h5file = h5.File(str(file))
    data = h5file["pressure"]
    time = h5file["t-coordinate"][:-1]

    fig, ax = plt.subplots(1, 2)
    varrt = np.zeros((data.shape[0],))
    boprt = np.zeros((data.shape[0],))

    varprodmd_mean_mrse = np.zeros_like(time)
    bopdmd_mean_mrse = np.zeros_like(varprodmd_mean_mrse)

    var_hat_varprodmd = np.zeros_like(varprodmd_mean_mrse)
    var_hat_bopdmd = np.zeros_like(bopdmd_mean_mrse)

    for run, (bopdmd, vardmd) in tqdm(
        enumerate(test_3dcfd(data, time, split)), total=data.shape[0]
    ):
        delta_bopdmd = bopdmd["mrse"] - bopdmd_mean_mrse
        delta_vardmd = vardmd["mrse"] - varprodmd_mean_mrse
        bopdmd_mean_mrse += delta_bopdmd / float(run + 1)
        varprodmd_mean_mrse += delta_vardmd / float(run + 1)
        var_hat_bopdmd += (bopdmd["mrse"] - bopdmd_mean_mrse) * delta_bopdmd
        var_hat_varprodmd += (vardmd["mrse"] - varprodmd_mean_mrse) * delta_vardmd
        varrt[run] = vardmd["dt"]
        boprt[run] = bopdmd["dt"]

    std_varprodmd = np.sqrt(var_hat_varprodmd / float(data.shape[0] - 1))
    std_bopdmd = np.sqrt(var_hat_bopdmd / float(data.shape[0] - 1))

    # plot expected errors and 95 % confidence interval
    ax[0].plot(time, bopdmd_mean_mrse, color="r", label="BOPDMD")
    ax[0].plot(time, varprodmd_mean_mrse, "--", color="b", label="VarProDMD")
    ax[0].fill_between(
        time,
        bopdmd_mean_mrse - 1.96 * std_bopdmd,
        bopdmd_mean_mrse + 1.96 * std_bopdmd,
        alpha=0.2,
        color="r",
    )
    ax[0].fill_between(
        time,
        varprodmd_mean_mrse - 1.96 * std_varprodmd,
        varprodmd_mean_mrse + 1.96 * std_varprodmd,
        alpha=0.2,
        color="b",
    )
    rect = Rectangle(
        (0, ax[0].get_ylim()[0]),
        time[int((1.0 - split) * data.shape[1])],
        np.subtract(*ax[0].get_ylim()[::-1]),
        facecolor="grey",
        alpha=0.4,
    )
    ax[0].add_patch(rect)
    ax[0].set_xlim(time[0], time[-1])
    ax[0].set_xlabel("t")
    ax[0].set_ylabel("RMSE")
    ax[0].legend()
    ax[0].grid()
    ax[1].violinplot([varrt, boprt], [1, 2], showmeans=True)
    ax[1].set_xticks([1, 2], labels=["VarProDMD", "BOPDMD"])
    ax[1].set_ylabel("t")
    ax[1].grid()
    plt.show()
