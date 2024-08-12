"""
VarProDMD vs BOPDMD on MRSE
"""
from __future__ import annotations

import argparse
import inspect
import logging
import pickle
from itertools import product, starmap
from pathlib import Path
from typing import Any

import numpy as np

from varprodmdstatspy.util.experiment_utils import (
    comp_checker,
    dmd_stats,
    signal,
    std_checker,
)

logging.basicConfig(level=logging.INFO, filename=__name__)

OPT_ARGS = {"method": "trf", "tr_solver": "exact", "loss": "linear"}


def test_high_dim_signal(
    method: str, n_runs: int, std: float, eps: float
) -> dict[str, Any]:
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)
    z = signal(__x, __time).T

    mean_err, mean_dt, c_xx, c_xy, c_yy = dmd_stats(
        method, z, time, std, OPT_ARGS, eps, n_iter=n_runs
    )
    return {
        "case": "High dimensional signal",
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


def run_mrse():
    STD = [0, 1e-4, 1e-3, 1e-2]
    N_RUNS = 100
    COMPS = [0.0, 0.2, 0.4, 0.6, 0.8]

    currentdir = Path(inspect.getfile(inspect.currentframe())).resolve().parent

    OUTDIR = currentdir / "output"
    parser = argparse.ArgumentParser("VarProDMD vs BOPDMD stats")

    parser.add_argument(
        "-c",
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
        "-l",
        "--loss",
        dest="loss",
        default="linear",
        type=str,
        help='Loss for optimization. Only useful if opt is not set to"lm" [Default: linear]',
    )
    parser.add_argument(
        "--opt",
        dest="opt",
        default="trf",
        type=str,
        help="Optimizer, [Default: trf - Trust Region Function]",
    )
    parser.add_argument(
        "--scale_jac",
        action="store_true",
        default=False,
        dest="scale_jac",
        help="Scale the search directions with inverse jacobian, [Default: False]",
    )
    __args = parser.parse_args()

    if __args.scale_jac:
        OPT_ARGS["x_scale"] = "jac"

    if __args.opt == "lm":
        OPT_ARGS["method"] = "lm"
        OPT_ARGS["loss"] = "linear"
        OPT_ARGS.pop("tr_solver", None)

    N_RUNS = abs(__args.runs)
    STD = __args.std
    COMPS = __args.compression
    OPT_ARGS["loss"] = __args.loss
    logging.info("Solver parameters")
    logging.info("=================")
    for kw in OPT_ARGS:
        msg = f"{kw}: {OPT_ARGS[kw]}"
        logging.info(msg)
    logging.info("\nStarting simulation...")

    args_in = []
    for comp, std in product(COMPS, STD):
        args_in.append(("VarProDMD", N_RUNS, std, comp))

    for std in STD:
        args_in.append(("BOPDMD", N_RUNS, std, 0))

    comp_list = []
    method_list = []
    exec_time_mean_list = []
    c_xx_list = []
    c_xy_list = []
    c_yy_list = []
    std_noise_list = []
    mrse_mean_list = []

    for res in starmap(test_high_dim_signal, args_in):
        logging.info(res["case"])

        std = res["std"]
        method = res["method"]
        mean_mrse = res["mean_err"]
        mean_t = res["mean_dt"]
        std_t = np.sqrt(res["c_yy"])
        comp_list.append(res["compression"] if res["compression"] > 0 else 0)
        method_list.append(method)
        exec_time_mean_list.append(mean_t)
        c_xx_list.append(res["c_xx"])
        c_xy_list.append(res["c_xy"])
        c_yy_list.append(res["c_yy"])
        std_noise_list.append(std)
        mrse_mean_list.append(mean_mrse)

        msg = f"{method} - Mean RSE: {mean_mrse}"
        logging.info(msg)

        stats = " ".join(
            [f"{method} - Mean exec time: {mean_t} [s],", f"Std exec time: {std_t} [s]"]
        )
        logging.info(stats)

        if std > 0:
            msg = f"{method} - Noise STD: {std}"
            logging.info(msg)

        if method == "VarProDMD":
            comp = res["compression"]
            if comp > 0:
                msg = f"VarProDMD compression: {comp * 100:.2f}%"
                logging.info(msg)
        logging.info("\n")

    data_dict = {
        "Method": method_list,
        "c": comp_list,
        "E[t]": exec_time_mean_list,
        "E[MRSE]": mrse_mean_list,
        "STD_NOISE": std_noise_list,
        "c_xx": c_xx_list,
        "c_xy": c_xy_list,
        "c_yy": c_yy_list,
    }
    loss = OPT_ARGS["loss"]
    opt = OPT_ARGS["method"]
    out_path = Path(__args.out) / opt

    if not Path(out_path).exists():
        Path(out_path).mkdir(parents=True)

    if "x_scale" not in OPT_ARGS:
        FILE_OUT = out_path / f"MRSE_highdim_{N_RUNS}_{loss}.pkl"
    else:
        FILE_OUT = out_path / f"MRSE_highdim_{N_RUNS}_{loss}_inv_jac_scale.pkl"

    msg = f"Storing results to {FILE_OUT}"
    logging.info(msg)

    with Path(FILE_OUT).open("wb") as handle:
        pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    run_mrse()
