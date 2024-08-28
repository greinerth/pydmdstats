"""
Conduct 3D fluid dynamic example taken from pdebench data
"""
from __future__ import annotations

import argparse
import inspect
import pickle
import timeit
from pathlib import Path
from typing import Any

import h5py as h5
import numpy as np
from pydmd import BOPDMD, VarProDMD
from tqdm import tqdm


def test_3dcfd(
    data: np.ndarray,
    time: np.ndarray,
    split: float = 0.3,
    compression: float = 0.0,
    loss: str = "linear",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Test 3D CFD example

       Split data into training set and test model on unseen timesteps (extrapolation).
       In addition record the time required for optimization.

    :param data: Current trajectory of shape [n, sx, sy, sz, 3]
    :type path2data: str
    :param split: Splitting parameter to split data to training- and test set, defaults to 0.7
    :type split: float, optional
    :param compression: Compression for VarProDMD.
        If `compression=0` no compression is performed, defaults to 0.0
    :type compression: float, optional
    :param loss: Loss for VarProDMD. Can be "linear", "huber", "soft_l1", "cauchy", "arctan"
    :raises ValueError: If parameter `split` not in (0, 1)
    :raises ValueError: If parameter `data` is not of shape [n, sy, sy, sz, 3]
    :return: List of results of each trial, time and number of training samples
    :rtype: tuple[dict[str, Any], dict[str, Any]]
    """
    if not 0.0 < split < 1.0:
        msg = "'split' must lie in (0, 1)!"
        raise ValueError(msg)

    if data.shape[-1] != 3 or data.ndim != 5:
        msg = "Expected 5D array of shape [n, sx, sy, sz, 3]"
        raise ValueError(msg)

    n_train = int((1.0 - split) * time.shape[-1])
    dataflat = np.zeros((np.prod(data.shape[1:]), data.shape[0]))

    for i in range(data.shape[0]):
        dataflat[:, i] = np.ravel(data[i, ...], "F")

    vardmd = VarProDMD(
        optargs={
            "method": "trf",
            "tr_solver": "exact",
            "loss": loss,
            "x_scale": "jac",
        },
        compression=abs(compression),
    )

    bopdmd = BOPDMD(trial_size=n_train)

    t0 = timeit.default_timer()
    bopdmd.fit(dataflat[:, :n_train], time[:n_train])
    dtbop = timeit.default_timer() - t0

    t0 = timeit.default_timer()
    vardmd.fit(dataflat[:, :n_train], time[:n_train])
    dtvar = timeit.default_timer() - t0

    bopnmrse = (
        np.linalg.norm(dataflat - bopdmd.forecast(time), axis=0)
        / np.sqrt(dataflat.shape[0])
        / np.std(dataflat, axis=0)
    )
    varnmrse = (
        np.linalg.norm(dataflat - vardmd.forecast(time), axis=0)
        / np.sqrt(dataflat.shape[0])
        / np.std(dataflat, axis=0)
    )

    return (
        {
            "method": "BOPDMD",
            "dt": dtbop,
            "nmrse": bopnmrse,
        },
        {
            "method": "VarPorDMD",
            "dt": dtvar,
            "nmrse": varnmrse,
        },
    )


def run_3dcfd() -> None:
    """Run 3DCFD experiment

    :raises FileNotFoundError: If path to dataset specified within the argparser
        does not exist.
    """
    currentdir = Path(inspect.getfile(inspect.currentframe())).resolve().parent
    outdir = currentdir / "output"
    parser = argparse.ArgumentParser(
        description="Perform experiments on 3D Fluid Dynamics dataset provided by pdebench repository."
    )
    parser.add_argument(
        "-s",
        "--split",
        type=float,
        default=0.3,
        dest="split",
        help="<Optional> Split a recorded trajectory for training and extraplotation. [Default: 0.3]",
    )
    parser.add_argument(
        "-n",
        "--ntrials",
        type=int,
        default=-1,
        dest="ntrials",
        help="<Optional> Specify the number of experinemts to run. If negative all experiments are used. [Default: -1]",
    )
    parser.add_argument(
        "-c",
        "--compression",
        type=float,
        default=0.0,
        dest="compression",
        help="<Optional> Compression for VarProDMD. If set to 0, VarProDMD will not perform compression. [Default: 0.0]",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        dest="data",
        help="<Required> Specify path to .hdf5 data file",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=outdir,
        dest="out",
        help=f"Output Directory. [Defaults: {outdir}]",
    )
    lh = ",".join(['"linear"', '"soft_l1"', '"huber"', '"cauchy"', '"arctan"'])
    lh = " ".join(
        [
            rf"<Optional> Specify the loss (e.g. {lh}) for robust regression.",
            '[Default: "linear"]',
        ]
    )
    parser.add_argument(
        "-l", "--loss", type=str, dest="loss", default="linear", help=lh
    )
    args = parser.parse_args()

    if not Path(args.data).exists():
        msg = f"{args.data} does not exist!"
        raise FileNotFoundError(msg)

    h5file = h5.File(str(args.data), "r")

    time = h5file["t-coordinate"][:-1].astype(np.float64)
    n_samples = (
        h5file["omega_x"].shape[0]
        if args.ntrials <= 0
        else min(args.ntrials, h5file["omega_x"].shape[0])
    )

    varrt = np.zeros((n_samples,))
    boprt = np.zeros((n_samples,))

    varprodmd_mean_nrmse = np.zeros_like(time)
    bopdmd_mean_nrmse = np.zeros_like(varprodmd_mean_nrmse)

    var_hat_varprodmd = np.zeros_like(varprodmd_mean_nrmse)
    var_hat_bopdmd = np.zeros_like(bopdmd_mean_nrmse)

    for run in tqdm(range(n_samples)):
        omega_x = h5file["omega_x"][run][:]
        omega_y = h5file["omega_y"][run][:]
        omega_z = h5file["omega_z"][run][:]
        data_in = np.concatenate(
            [omega_x[..., None], omega_y[..., None], omega_z[..., None]], axis=-1
        )
        bopdmd, vardmd = test_3dcfd(
            data_in, time, args.split, args.compression, args.loss
        )

        # calculate running average and variance
        delta_bopdmd = bopdmd["nmrse"] - bopdmd_mean_nrmse
        delta_vardmd = vardmd["nmrse"] - varprodmd_mean_nrmse
        bopdmd_mean_nrmse += delta_bopdmd / float(run + 1)
        varprodmd_mean_nrmse += delta_vardmd / float(run + 1)
        var_hat_bopdmd += (bopdmd["nmrse"] - bopdmd_mean_nrmse) * delta_bopdmd
        var_hat_varprodmd += (vardmd["nmrse"] - varprodmd_mean_nrmse) * delta_vardmd
        varrt[run] = vardmd["dt"]
        boprt[run] = bopdmd["dt"]

    std_varprodmd = np.sqrt(var_hat_varprodmd / float(n_samples - 1))
    std_bopdmd = np.sqrt(var_hat_bopdmd / float(n_samples - 1))

    data_out = {
        "time": time,
        "mean-nRMSE-VarProDMD": varprodmd_mean_nrmse,
        "mean-nRMSE-BOPDMD": bopdmd_mean_nrmse,
        "std-nRMSE-VarProDMD": std_varprodmd,
        "std-nRMSE-BOPDMD": std_bopdmd,
        "dt-VarProDMD": varrt,
        "dt-BOPDMD": boprt,
    }

    out_path = Path(args.out) / "trf"
    fname = args.data.split("/")[-1]
    fname = fname.split(".hdf5")[0]
    out_path = out_path / fname

    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    file_out = (
        out_path
        / f"nRMSE_{n_samples}_runs_{args.split}_split_{
        args.compression}_comp.pkl"
    )

    with Path(file_out).open("wb") as handle:
        pickle.dump(data_out, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_3dcfd()
