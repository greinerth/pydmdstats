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
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD

from varprodmdstatspy.util.experiment_utils import (
    OPT_ARGS,
    comp_checker,
    exec_times_bop_dmd,
    exec_times_varpro_dmd,
    signal2d,
    ssim_multi_images,
    std_checker,
)

logging.basicConfig(level=logging.INFO)

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
    snapshots_flat = np.zeros((np.prod(data[0].shape), len(data)), dtype=complex)
    for i in range(len(data)):
        __img = data[i].copy()
        if std > 0:
            __img += np.random.normal(0, std, data[i].shape)
        snapshots_flat[:, i] = np.ravel(__img)
    if method == "VarProDMD":
        __dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)

    elif method == "BOPDMD":
        __dmd = BOPDMD()

    else:
        raise ValueError(f"{method} not implemented")

    __dmd.fit(snapshots_flat, time)
    omega_size = __dmd.eigs.size
    __dmd_pred = __dmd.forecast(time)
    dmd_rec = []
    data_in = []
    for i in range(snapshots_flat.shape[-1]):
        __dmd_rec = __dmd_pred[:, i].reshape(data[0].shape)
        dmd_rec.append(np.concatenate([__dmd_rec.real, __dmd_rec.imag], axis=-1))
        data_in.append(np.concatenate([data[i].real, data[i].imag], axis=-1))
    dmd_rec = np.concatenate(dmd_rec, axis=0)
    data_in = np.concatenate(data_in, axis=0)
    mean, var = ssim_multi_images(data_in, dmd_rec)
    del data_in, data, dmd_rec

    __stats = (
        exec_times_bop_dmd(snapshots_flat, time, n_runs)
        if method == "BOPDMD"
        else exec_times_varpro_dmd(snapshots_flat, time, eps, OPT_ARGS, n_runs)
    )

    return {
        "case": "Complex 2D signal",
        "omega_size": omega_size,
        "method": method,
        "SSIM": (mean, var),
        "compression": eps,
        "n_runs": n_runs,
        "stats": __stats,
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
    noisy_imgs = np.zeros_like(imgs)
    mean: float = 0.0
    var: float = 0.0
    omega_size: int = 0
    for i in range(time.size):
        imgs[i] = signal2d(x, y, x_0, y_0, velocity, time[i])
        noisy_imgs[i] = imgs[i].copy()
        if std > 0:
            noisy_imgs[i] += np.random.normal(0, std, noisy_imgs[0].shape)

    imgs = np.expand_dims(imgs, axis=-1)
    __flat = np.zeros((128 * 128, time.size))
    for i in range(imgs.shape[0]):
        __flat[:, i] = noisy_imgs[i].ravel()
    __flat = __flat.astype(np.complex128)

    if method == "VarProDMD":
        __dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)

    elif method == "BOPDMD":
        __dmd = BOPDMD()

    else:
        raise ValueError(f"{method} not implemented")

    __dmd.fit(__flat, time)

    del noisy_imgs

    omega_size = __dmd.eigs.size
    __dmd_pred = __dmd.forecast(time)
    dmd_rec = []

    for i in range(__flat.shape[-1]):
        __dmd_rec = np.expand_dims(__dmd_pred[:, i].reshape(imgs.shape[1:]), axis=0)
        dmd_rec.append(__dmd_rec.real)
    dmd_rec = np.concatenate(dmd_rec, axis=0)
    mean, var = ssim_multi_images(imgs, dmd_rec)

    del imgs, dmd_rec

    __stats = (
        exec_times_bop_dmd(__flat, time, n_runs)
        if method == "BOPDMD"
        else exec_times_varpro_dmd(__flat, time, eps, OPT_ARGS, n_runs)
    )

    return {
        "case": "Moving points",
        "omega_size": omega_size,
        "method": method,
        "SSIM": (mean, var),
        "compression": eps,
        "n_runs": n_runs,
        "stats": __stats,
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
    dt = YEARS / float(n_samples)
    sst = sst[-128:]
    mean = 0
    var = 0
    time = np.arange(sst.shape[0]) * dt
    img0 = sst[0]
    # img0 = img0[::-1, ::]
    msk = ~(img0 < low)
    msk &= ~(img0 > high)

    # remember invalid numbers from flipped original image
    # need to invert once again else not the correct values are remembered
    rows, cols = np.where(~msk)
    msk = np.ravel(msk)
    __flat = np.zeros(shape=(np.sum(msk), sst.shape[0]))
    for i in range(sst.shape[0]):
        __img = sst[i]
        __noisy = __img.copy()
        if std > 0:
            __noisy += np.random.normal(0.0, std, size=__img.shape)
        __flat[:, i] = np.ravel(__noisy)[msk]
    __flat = __flat.astype(np.complex128)
    if method == "VarProDMD":
        __dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)

    elif method == "BOPDMD":
        __dmd = BOPDMD()
    else:
        raise ValueError(f"{method} not implemented")

    __dmd.fit(__flat, time)
    omega_size = __dmd.eigs.size
    __dmd_pred = __dmd.forecast(time).real
    dmd_rec = []
    for i in range(__flat.shape[-1]):
        __flat_in = np.zeros((msk.shape[-1],))
        __flat_in[msk] = __dmd_pred[:, i].real
        __dmd_rec = np.expand_dims(__flat_in.reshape([1] + list(sst[0].shape)), axis=-1)
        dmd_rec.append(__dmd_rec)
    dmd_rec = np.concatenate(dmd_rec, axis=0)
    # sst = sst[:, ::-1, :]
    sst[:, rows, cols] = 0.0
    mean, var = ssim_multi_images(np.expand_dims(sst, axis=-1), dmd_rec)

    del sst, ds, dmd_rec

    __stats = (
        exec_times_bop_dmd(__flat, time, n_runs)
        if method == "BOPDMD"
        else exec_times_varpro_dmd(__flat, time, eps, OPT_ARGS, n_runs)
    )

    return {
        "case": "Global temperature",
        "omega_size": omega_size,
        "method": method,
        "SSIM": (mean, var),
        "compression": eps,
        "n_runs": n_runs,
        "stats": __stats,
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
    COMPS = [0, 0.4, 0.6, 0.8]
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

    results = list(starmap(fcts[__args.fct], __args_in))

    comp_list = []
    method_list = []
    exec_time_mean_list = []
    exec_time_std_list = []
    noise_std = []
    case_list = []
    omega_list = []
    ssim_mean_list = []
    ssim_var_list = []

    for res in results:
        logging.info(Fore.CYAN + res["case"])
        method = res["method"]
        omega_size = res["omega_size"]
        mean_t = res["stats"].mean
        var_t = res["stats"].var
        __std = res["std"]
        comp_list.append(res["compression"] if res["compression"] > 0 else 0)
        method_list.append(method)
        exec_time_mean_list.append(mean_t)
        exec_time_std_list.append(np.sqrt(var_t))
        case_list.append(res["case"])
        omega_list.append(omega_size)
        mean, var = res["SSIM"]
        ssim_mean_list.append(mean)
        ssim_var_list.append(var)
        noise_std.append(__std)

        logging.info(Fore.WHITE + f"{method} - Mean SSIM: {mean}, Var SSIM: {var}")
        logging.info(Fore.WHITE + f"{method} - OMEGAS: {omega_size}")
        stats = "{} - Mean exec time: {} [s], Std exec time: {} [s]"
        logging.info(Fore.WHITE + stats.format(method, mean_t, np.sqrt(var_t)))
        if __std > 0:
            logging.info(Fore.WHITE + f"{method} - Noise STD: {__std}")
        if method == "VarProDMD":
            comp = res["compression"]
            if comp > 0:
                logging.info(Fore.WHITE + f"VarProDMD compression: {comp * 100:.2f}%")
        logging.info("\n")

    data_dict = {
        "Method": method_list,
        "N_eigs": omega_list,
        "c": comp_list,
        "Experiment": case_list,
        "E[t]": exec_time_mean_list,
        "E[SSIM]": ssim_mean_list,
        "SSIM_VAR": ssim_var_list,
        "STD_RUNTIME": exec_time_std_list,
        "STD_NOISE": noise_std,
        "N_RUNS": N_RUNS,
    }
    FILE_OUT = os.path.join(__args.out, f"SSIM_{__args.fct}_{N_RUNS}.pkl")

    logging.info(f"Storing results to {FILE_OUT}")
    with open(FILE_OUT, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_ssim()
