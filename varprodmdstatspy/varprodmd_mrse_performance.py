#!/usr/bin/python3
# pylint: skip-file
"""
VarProDMD vs BOPDMD on MRSE
"""

import argparse
import inspect
import logging
import os
import pickle
from itertools import product, starmap
from typing import Any, Dict

import numpy as np
from colorama import Fore
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD

from varprodmdstatspy.util.experiment_utils import (
    comp_checker,
    dmd_stats,
    signal,
    std_checker,
)

logging.basicConfig(level=logging.INFO, filename=__name__)
# logging.root.setLevel(logging.INFO)
OPT_ARGS = {"method": 'trf', "tr_solver": 'exact', "loss": 'linear'}

def test_high_dim_signal(
    method: str, n_runs: int, std: float, eps: float
) -> Dict[str, Any]:
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)
    z = signal(__x, __time).T

    if method == "VarProDMD":
        dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)

    elif method == "BOPDMD":
        dmd = BOPDMD()
    else:
        raise ValueError(f"{method} not implemented")

    time_stats, error_stats = dmd_stats(dmd, z, time, std, n_iter=n_runs)

    return {
        "case": "High dimensional signal",
        # "omega_size": omega_size,
        "method": method,
        "compression": eps,
        "n_runs": n_runs,
        "time_stats": time_stats,
        "error_stats": error_stats,
        "std": std,
    }


def run_mrse():
    STD = [0, 1e-4, 1e-3, 1e-2]
    N_RUNS = 100
    COMPS = [0.0, 0.2, 0.4, 0.6, 0.8]

    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )

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

    __args = parser.parse_args()
    # manager = mp.Manager()
    # results = manager.list()

    if not os.path.exists(__args.out):
        os.makedirs(__args.out)

    N_RUNS = abs(__args.runs)
    STD = __args.std
    COMPS = __args.compression

    logging.info("Solver parameters")
    logging.info("=================")
    logging.info("\nStarting simulation...")
    
    args_in = []
    for comp, std in product(COMPS, STD):
        args_in.append(("VarProDMD", N_RUNS, std, comp))

    for std in STD:
        args_in.append(("BOPDMD", N_RUNS, std, 0))

    comp_list = []
    method_list = []
    exec_time_mean_list = []
    exec_time_std_list = []
    std_noise_list = []
    # omega_list = []
    mrse_mean_list = []
    mrse_std_list = []

    for res in starmap(test_high_dim_signal, args_in):
        logging.info(Fore.CYAN + res["case"])
        method = res["method"]
        # omega_size = res["omega_size"]
        mean_t = res["time_stats"].mean
        try:
            std_t = res["time_stats"].std
        except ZeroDivisionError:
            std_t = 0.0

        __std = res["std"]
        comp_list.append(res["compression"] if res["compression"] > 0 else 0)
        method_list.append(method)
        exec_time_mean_list.append(mean_t)
        exec_time_std_list.append(std_t)
        # case_list.append(res["case"])
        # omega_list.append(omega_size)
        std_noise_list.append(__std)
        mean_mrse = res["error_stats"].mean
        try:
            std_mrse = res["error_stats"].std
        except ZeroDivisionError:
            std_mrse = 0.0

        mrse_mean_list.append(mean_mrse)
        mrse_std_list.append(std_mrse)

        logging.info(Fore.WHITE + f"{method} - Mean RSE: {mean_mrse}")
        # logging.info(Fore.WHITE + f"{method} - OMEGAS: {omega_size}")
        stats = "{} - Mean exec time: {} [s], Std exec time: {} [s]"
        logging.info(Fore.WHITE + stats.format(method, mean_t, std_t))

        if __std > 0:
            logging.info(Fore.WHITE + f"{method} - Noise STD: {__std}")

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
        "E[MRSE]": mrse_mean_list,
        "std[MRSE]": mrse_std_list,
        "STD_NOISE": std_noise_list,
        "STD_RUNTIME": exec_time_std_list,
        # "N_RUNS": N_RUNS,
    }
    loss = OPT_ARGS["loss"]
    FILE_OUT = os.path.join(__args.out, f"MRSE_highdim_{N_RUNS}_{loss}.pkl")
    logging.info(f"Storing results to {FILE_OUT}")
    with open(FILE_OUT, "wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_mrse()
