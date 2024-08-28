"""Visualize the results of BOPDMD vs VarProDMD on 3DCFD dataset"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from matplotlib.patches import Rectangle


def visualize_results():
    """Visualize results of 3DCFD experiment

    :raises FileExistsError: When file specified within the argparser does not exist.
    """

    dpi = 300.0
    # height = 531.0
    # width = 1200.0

    plt.style.use("science")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.dpi"] = dpi

    parser = argparse.ArgumentParser("Visualize 3DCFD Results")
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        type=str,
        required=True,
        help="Path to .pkl file",
    )

    parser.add_argument(
        "--conf",
        type=float,
        dest="conf",
        default=1.96,
        help="Factor for premultiplying the standard deviations (confidence). [Default: 1.96]",
    )

    args = parser.parse_args()
    nstd = abs(args.conf)

    if not Path(args.path).exists():
        msg = f"{args.path} does not exist!"
        raise FileExistsError(msg)

    with Path(args.path).open("rb") as handle:
        data = pickle.load(handle)
        fname = args.path.split("/")[-1]
        fname = fname.split(".pkl")[0]
        metadata = fname.split("_")
        splitcoeff = float(metadata[3])

        plt.rcParams["figure.dpi"] = dpi
        plt.style.use("science")
        _, ax = plt.subplots(1, 2, layout="constrained")

        # plot expected errors and 95 % confidence interval
        ax[0].plot(data["time"], data["mean-nRMSE-BOPDMD"], color="r", label="BOPDMD")
        ax[0].plot(
            data["time"],
            data["mean-nRMSE-VarProDMD"],
            "--",
            color="b",
            label="VarProDMD",
        )
        ax[0].fill_between(
            data["time"],
            data["mean-nRMSE-BOPDMD"] - nstd * data["std-nRMSE-BOPDMD"],
            data["mean-nRMSE-BOPDMD"] + nstd * data["std-nRMSE-BOPDMD"],
            alpha=0.2,
            color="r",
        )
        ax[0].fill_between(
            data["time"],
            data["mean-nRMSE-VarProDMD"] - nstd * data["std-nRMSE-VarProDMD"],
            data["mean-nRMSE-VarProDMD"] + nstd * data["std-nRMSE-VarProDMD"],
            alpha=0.2,
            color="b",
        )
        rect = Rectangle(
            (0, ax[0].get_ylim()[0]),
            data["time"][int((1.0 - splitcoeff) * data["time"].shape[-1])],
            np.subtract(*ax[0].get_ylim()[::-1]),
            facecolor="grey",
            alpha=0.4,
        )
        ax[0].add_patch(rect)
        ax[0].set_xlim(data["time"][0], data["time"][-1])
        ax[0].set_xlabel(r"$t$ in $s$")
        ax[0].set_ylabel("nRMSE")
        ax[0].set_title("a) Extrapolation", loc="left", fontsize=8)
        ax[0].legend()
        ax[0].grid()
        ax[1].violinplot(
            [data["dt-VarProDMD"], data["dt-BOPDMD"]], [1, 2], showmeans=True
        )
        ax[1].set_xticks([1, 2], labels=["VarProDMD", "BOPDMD"])
        ax[1].set_ylabel(r"$t$ in $s$")
        ax[1].set_title("b) Runtimes", loc="left", fontsize=8)
        ax[1].grid()
        # fig.set_size_inches(width / dpi, height / dpi)
        plt.show()


if __name__ == "__main__":
    visualize_results()
