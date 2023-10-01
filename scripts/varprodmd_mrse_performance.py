#!/usr/bin/python3
# pylint: skip-file
"""
VarProDMD vs BOPDMD on MRSE
"""
import argparse
import inspect
import os
import pickle
from itertools import product, starmap
from typing import Any, Dict

import numpy as np
from colorama import Fore
from scripts.util.experiment_utils import (OPT_ARGS, comp_checker, exec_times_bop_dmd,
                                   exec_times_varpro_dmd, signal, std_checker)
from pydmd.bopdmd import BOPDMD
from pydmd.varprodmd import VarProDMD


def test_high_dim_signal(method: str,
                         n_runs: int,
                         std: float,
                         eps: float) -> Dict[str, Any]:
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)
    z = signal(__x, __time).T
    z_in = z.copy()
    if std > 0:
        z_in += np.random.normal(0., std, size=z.shape)
    # __idx = select_best_samples(z, 2.5e-14)
    # __idx = select_best_samples_fast(z, 0.1)[0]
    if method == "VarProDMD":
        __dmd = VarProDMD(compression=eps, optargs=OPT_ARGS)

    elif method == "BOPDMD":
        __dmd = BOPDMD()
    else:
        raise ValueError(f"{method} not implemented")

    __dmd.fit(z_in, time)
    omega_size = __dmd.eigs.size
    dmd_pred = __dmd.forecast(time)
    mrse_dmd = np.linalg.norm(z - dmd_pred) / np.sqrt(z.shape[-1])
    del z

    __stats = exec_times_bop_dmd(z_in, time, n_runs) if method == "BOPDMD" \
        else exec_times_varpro_dmd(z_in, time, eps, OPT_ARGS, n_runs)

    return {"case": "High dimensional signal",
            "omega_size": omega_size,
            "method": method,
            "MRSE": mrse_dmd,
            "compression": eps,
            "n_runs": n_runs,
            "stats": __stats,
            "std": std}


def run_mrse():

    STD = [0, 1e-4, 1e-3, 1e-2]
    N_RUNS = 100
    COMPS = [0, 0.2, 0.4, 0.6, 0.8]

    currentdir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))

    # FILE = os.path.join(PATH, DATASET)
    OUTDIR = os.path.join(currentdir, "output")
    parser = argparse.ArgumentParser("VarProDMD vs BOPDMD stats")

    parser.add_argument("--compression",
                        metavar='N',
                        nargs="+",
                        type=comp_checker,
                        default=COMPS,
                        dest="compression",
                        help=f"Compression for VarProDMD. [Defaults: {COMPS}]")
    parser.add_argument("-s",
                        "--std",
                        metavar='N',
                        type=std_checker, nargs='+',
                        dest="std",
                        default=STD, help=f"Standard Deviation for noise. [Defaults: {STD}]")
    parser.add_argument("-o",
                        "--out",
                        type=str,
                        default=OUTDIR,
                        dest="out",
                        help=f"Output Directory. [Defaults: {OUTDIR}]")
    parser.add_argument("-r",
                        "--runs",
                        type=int,
                        default=N_RUNS,
                        dest="runs",
                        help=f"Number of runs per configuration [Defaults: {N_RUNS}]")
    __args = parser.parse_args()
    # manager = mp.Manager()
    # results = manager.list()
    results = []
    if not os.path.exists(__args.out):
        os.makedirs(__args.out)

    N_RUNS = abs(__args.runs)
    STD = __args.std
    COMPS = __args.compression

    print("Starting simulation...")
    __args_in = []
    for comp, std in product(COMPS, STD):
        __args_in.append(("VarProDMD", N_RUNS, std, comp))

    for std in STD:
        __args_in.append(("BOPDMD", N_RUNS, std, 0))

    results = list(starmap(test_high_dim_signal, __args_in))

    comp_list = []
    method_list = []
    exec_time_mean_list = []
    exec_time_std_list = []
    std_noise = []
    case_list = []
    omega_list = []
    mrse_mean_list = []

    for res in results:
        print(Fore.CYAN + res["case"])
        method = res["method"]
        omega_size = res["omega_size"]
        mean_t = res["stats"].mean
        var_t = res["stats"].var
        __std = res["std"]
        comp_list.append(res["compression"]
                         if res["compression"] > 0 else 0)
        method_list.append(method)
        exec_time_mean_list.append(mean_t)
        exec_time_std_list.append(np.sqrt(var_t))
        case_list.append(res["case"])
        omega_list.append(omega_size)
        std_noise.append(__std)
        mean = res["MRSE"]
        mrse_mean_list.append(mean)

        print(Fore.WHITE + f"{method} - Mean RSE: {mean}")
        print(
            Fore.WHITE + f"{method} - OMEGAS: {omega_size}")
        print(
            Fore.WHITE + f"{method} - Mean exec time: {mean_t} [s], Std exec time: {np.sqrt(var_t)} [s]")
        if __std > 0:
            print(
                Fore.WHITE + f"{method} - Noise STD: {__std}")
        if method == "VarProDMD":
            comp = res["compression"]
            if comp > 0:
                print(
                    Fore.WHITE + f"VarProDMD compression: {comp * 100:.2f}%")
        print("\n")

    data_dict = {"Method": method_list,
                 "N_eigs": omega_list,
                 "c": comp_list,
                 "Experiment": case_list,
                 "E[t]": exec_time_mean_list,
                 "E[RSE]": mrse_mean_list,
                 "STD_NOISE": std_noise,
                 "STD_RUNTIME": exec_time_std_list,
                 "N_RUNS": N_RUNS}

    FILE_OUT = os.path.join(__args.out, f"MRSE_highdim_{N_RUNS}.pkl")
    print(f"Storing results to {FILE_OUT}")
    with open(FILE_OUT, 'wb') as handle:
        pickle.dump(data_dict, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_mrse()
