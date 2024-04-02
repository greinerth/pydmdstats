#!/usr/bin/python3
# pylint: skip-file
"""
VarProDMD vs BOPDMD on SSIM
"""

import argparse
import inspect
import logging
import os
import pickle
from itertools import product, starmap
from typing import Any, Dict

# import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import wget
from colorama import Fore

from varprodmdstatspy.util.experiment_utils import (
    comp_checker,
    dmd_stats,
    dmd_stats_global_temp,
    signal2d,
    std_checker,
)

logging.basicConfig(level=logging.INFO, filename=__name__)
# logging.root.setLevel(logging.INFO)
OPT_ARGS = {"method": "trf", "tr_solver": "exact", "loss": "linear"}


# OPT_ARGS = {"method": 'lm', "loss": 'linear'}
def download(url: str, outdir: str):
    """Download dataset.
    Found on: https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads
    Args:
        url (str): url
        fname (str): Output
    """
    wget.download(url, outdir)


def test_complex2d_signal(
    method: str, n_runs: int, std: float, eps: float
) -> Dict[str, Any]:
    time = np.linspace(0, 6, 64)
    x1 = np.linspace(-3, 3, 128)
    x2 = np.linspace(-3, 3, 128)
    x1grid, x2grid = np.meshgrid(x1, x2)
    data = [
        np.expand_dims(
            np.array([2 / np.cosh(x1grid) / np.cosh(x2grid) * (1.2j**-t)]), axis=-1
        )
        for t in time
    ]

    data = np.concatenate(data, axis=0)
    mean_err, mean_dt, c_xx, c_xy, c_yy = dmd_stats(
        method, data, time, std, OPT_ARGS, eps, n_iter=n_runs
    )
    return {
        "case": "Complex 2D signal",
        # "omega_size": omega_size,
        "method": method,
        "compression": eps,
        "n_runs": n_runs,
        "mean_err": mean_err,
        "mean_dt": mean_dt,
        "c_xx": c_xx,
        "c_xy": c_xy,
        "c_yy": c_yy,
        "std": std,
    }


def test_2_moving_points(
    method: str, n_runs: int, std: float, eps: float
) -> Dict[str, Any]:
    fps = 30.0
    total_time = 5.0
    velocity = fps / total_time / 4
    x_0 = np.array([-18, -20])
    y_0 = np.array([20, -9])
    time = np.linspace(0, 5, 128)
    x, y = np.meshgrid(np.arange(-64, 64), np.arange(-64, 64))
    imgs = np.zeros((time.size, 128, 128))

    for i in range(time.size):
        imgs[i] = signal2d(x, y, x_0, y_0, velocity, time[i])

    imgs = np.expand_dims(imgs, axis=-1)
    mean_err, mean_dt, c_xx, c_xy, c_yy = dmd_stats(
        method, imgs, time, std, OPT_ARGS, eps, n_iter=n_runs
    )
    return {
        "case": "Moving points",
        # "omega_size": omega_size,
        "method": method,
        "compression": eps,
        "n_runs": n_runs,
        "mean_err": mean_err,
        "mean_dt": mean_dt,
        "c_xx": c_xx,
        "c_xy": c_xy,
        "c_yy": c_yy,
        "std": std,
    }


def test_global_temp(
    method: str, n_runs: int, std: float, eps: float
) -> Dict[str, Any]:
    DATASET = "sst.day.mean.ltm.1982-2010.nc"
    YEARS = 2010 - 1982
    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    FILE = os.path.join(currentdir, "data")
    FILE = os.path.join(FILE, DATASET)
    ds = nc.Dataset(FILE)
    sst = ds["sst"][:]
    low, high = ds["sst"].valid_range
    n_samples = float(sst.shape[0])
    sst = sst[-128:]
    dt = (float(YEARS) / float(n_samples)) * sst.shape[0]

    time = np.linspace(0, dt, sst.shape[0])
    img0 = sst[0]
    # img0 = img0[::-1, ::]
    msk = ~(img0 < low)
    msk &= ~(img0 > high)

    mean_err, mean_dt, c_xx, c_xy, c_yy = dmd_stats_global_temp(
        method, sst, time, msk, std, OPT_ARGS, eps, n_runs
    )
    return {
        "case": "Global temperature",
        # "omega_size": omega_size,
        "method": method,
        "compression": eps,
        "n_runs": n_runs,
        "mean_err": mean_err,
        "mean_dt": mean_dt,
        "c_xx": c_xx,
        "c_xy": c_xy,
        "c_yy": c_yy,
        "std": std,
    }


def run_ssim():
    fcts = {
        "complex2d": test_complex2d_signal,
        "moving_points": test_2_moving_points,
        "global_temp": test_global_temp,
    }

    STD = [0, 1e-4, 1e-3, 1e-2]
    N_RUNS = 100
    COMPS = [0.0, 0.2, 0.4, 0.6, 0.8]
    FCTS = list(fcts.keys())

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )

    # PATH = os.path.join(currentdir, "data")
    # FILE = os.path.join(PATH, DATASET)
    OUTDIR = os.path.join(currentdir, "output")
    parser = argparse.ArgumentParser("VarProDMD vs BOPDMD stats")

    parser.add_argument(
        "--compression",
        metavar="N",
        nargs="+",
        type=comp_checker,
        default=COMPS,
        dest="compression",
        help=f"Compression for VarProDMD. [Defaults: {COMPS}]",
    )
    parser.add_argument(
        "-s",
        "--std",
        metavar="N",
        type=std_checker,
        nargs="+",
        dest="std",
        default=STD,
        help=f"Standard Deviation for noise. [Defaults: {STD}]",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=str,
        default=OUTDIR,
        dest="out",
        help=f"Output Directory. [Defaults: {OUTDIR}]",
    )
    parser.add_argument(
        "-r",
        "--runs",
        type=int,
        default=N_RUNS,
        dest="runs",
        help=f"Number of runs per configuration [Defaults: {N_RUNS}]",
    )
    parser.add_argument(
        "-f",
        "--function",
        required=True,
        dest="fct",
        type=str,
        help=f"Function to run: Available functions: {FCTS}",
    )

    __args = parser.parse_args()

    if __args.fct not in fcts:
        raise KeyError("f{__args.fct} not implemented!")
    PATH2DATASET = os.path.join(currentdir, "data")

    if __args.fct == "global_temp" and not os.path.exists(PATH2DATASET):
        logging.info("Downloading dataset...")
        os.makedirs(PATH2DATASET)
        download(
            "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1982-2010.nc",
            os.path.join(PATH2DATASET, "sst.day.mean.ltm.1982-2010.nc"),
        )

    if not os.path.exists(__args.out):
        os.makedirs(__args.out)

    N_RUNS = abs(__args.runs)
    STD = __args.std
    COMPS = __args.compression

    logging.info("Solver parameters")
    logging.info("=================")

    for key in OPT_ARGS:
        logging.info(f"{key}: {OPT_ARGS[key]}")

    logging.info("\nStarting simulation...")

    __args_in = []
    for comp, std in product(COMPS, STD):
        __args_in.append(("VarProDMD", N_RUNS, std, comp))

    for std in STD:
        __args_in.append(("BOPDMD", N_RUNS, std, 0))

    comp_list = []
    method_list = []
    exec_time_mean_list = []
    c_xx_list = []
    c_xy_list = []
    c_yy_list = []
    # exec_time_std_list = []
    std_noise_list = []
    # omega_list = []
    ssim_mean_list = []

    for res in starmap(fcts[__args.fct], __args_in):
        # logging.info(Fore.CYAN + res["case"])
        std = res["std"]
        method = res["method"]
        mean_ssim = res["mean_err"]
        mean_t = res["mean_dt"]
        std_t = np.sqrt(res["c_yy"])
        comp_list.append(res["compression"] if res["compression"] > 0 else 0)
        method_list.append(method)
        ssim_mean_list.append(mean_ssim)
        exec_time_mean_list.append(mean_t)
        c_xx_list.append(res["c_xx"])
        c_xy_list.append(res["c_xy"])
        c_yy_list.append(res["c_yy"])
        std_noise_list.append(std)
        # case_list.append(res["case"])
        # omega_list.append(omega_size)

        std_ssim = np.sqrt(res["c_xx"])
        logging.info(
            Fore.WHITE + f"{method} - Mean SSIM: {mean_ssim}, Std SSIM: {std_ssim}"
        )
        # logging.info(Fore.WHITE + f"{method} - OMEGAS: {omega_size}")
        stats = "{} - Mean exec time: {} [s], Std exec time: {} [s]"
        logging.info(Fore.WHITE + stats.format(method, mean_t, std_t))
        if std > 0:
            logging.info(Fore.WHITE + f"{method} - Noise STD: {std}")
        if method == "VarProDMD":
            comp = res["compression"]
            if comp > 0:
                logging.info(Fore.WHITE + f"VarProDMD compression: {comp * 100:.2f}%")
        logging.info("\n")

    data_dict = {
        "Method": method_list,
        # "N_eigs": omega_list,
        "c": comp_list,
        # "Experiment": case_list,
        "E[t]": exec_time_mean_list,
        "E[SSIM]": ssim_mean_list,
        "c_xx": c_xx_list,
        "c_xy": c_xy_list,
        "c_yy": c_yy_list,
        "STD_NOISE": std_noise_list,
        # "N_RUNS": N_RUNS,
    }
    loss = OPT_ARGS["loss"]
    FILE_OUT = os.path.join(__args.out, f"SSIM_{__args.fct}_{N_RUNS}_{loss}.pkl")

    logging.info(f"Storing results to {FILE_OUT}")
    with open(FILE_OUT, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_ssim()
