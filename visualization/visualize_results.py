"""Visualize the results of BOPDMD vs VarProDMD"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa: F401
import seaborn as sns
from matplotlib import transforms
from matplotlib.patches import Ellipse

logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)


def visualize_stats():
    """Visualize statistics

    :raises FileExistsError: When file does not exist
    :raises ValueError: When experiment is not supported.
    """
    plt.style.use("science")
    plt.rcParams["text.usetex"] = True
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser("Visualize Statistics")
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        type=str,
        required=True,
        help="Path to .pkl file",
    )
    __args = parser.parse_args()
    if not Path(__args.path).exists():
        msg = f"{__args.path} does not exist!"
        raise FileExistsError(msg)

    with Path(__args.path).open("rb") as handle:
        data = pickle.load(handle)

        dataframe = pd.DataFrame(data)
        filestr = f"\n{dataframe}"
        logging.info(filestr)
        std_noise = np.array(sorted(set(dataframe["STD_NOISE"])))

        if "E[SSIM]" in dataframe.columns:
            # std_error = dataframe["SSIM_STD"].to_numpy()
            expected_error = dataframe["E[SSIM]"].to_numpy()
            dataframe = dataframe.rename(
                {
                    "E[SSIM]": r"$E\left[\overline{SSIM}\right]$",
                    "E[t]": r"$E\left[t\right]$ in $s$",
                    "STD_NOISE": r"$\sigma_{std}$",
                    "c": r"$c_{comp}$",
                },
                axis="columns",
            )

            g0 = sns.FacetGrid(dataframe, col=r"$\sigma_{std}$")
            g0.map_dataframe(
                sns.scatterplot,
                r"$E\left[\overline{SSIM}\right]$",
                r"$E\left[t\right]$ in $s$",
                size="$c_{comp}$",
                legend="full",
                hue="Method",
                alpha=0.5,
            )

        elif "E[MRSE]" in dataframe.columns:
            # std_error = dataframe["std[MRSE]"].to_numpy()
            expected_error = dataframe["E[MRSE]"].to_numpy()
            dataframe = dataframe.rename(
                {
                    "E[MRSE]": r"$E\left[d\right]$ in $m$",
                    "E[t]": r"$E\left[t\right]$ in $s$",
                    "STD_NOISE": r"$\sigma_{std}$",
                    "c": r"$c_{comp}$",
                },
                axis="columns",
            )
            g0 = sns.FacetGrid(dataframe, col=r"$\sigma_{std}$")
            g0.map_dataframe(
                sns.scatterplot,
                r"$E\left[d\right]$ in $m$",
                r"$E\left[t\right]$ in $s$",
                size="$c_{comp}$",
                legend="full",
                hue="Method",
                alpha=0.5,
            )

        else:
            msg = "Unsupported Experiment!"
            raise ValueError(msg)

        axes = g0.axes.reshape((-1,))
        palette = sns.color_palette(n_colors=2)
        color = {"VarProDMD": palette[0], "BOPDMD": palette[1]}
        for ax, std in zip(axes, std_noise):
            rows = np.where(dataframe[r"$\sigma_{std}$"].to_numpy() == std)[0]
            rt_expected = dataframe[r"$E\left[t\right]$ in $s$"].to_numpy()[rows]
            cov_xx = dataframe["c_xx"].to_numpy()[rows]
            cov_xy = dataframe["c_xy"].to_numpy()[rows]
            cov_yy = dataframe["c_yy"].to_numpy()[rows]
            err_expected = expected_error[rows]
            algorithm = dataframe["Method"].to_numpy()[rows]
            for err, rt, c_xx, c_xy, c_yy, alg in zip(
                err_expected, rt_expected, cov_xx, cov_xy, cov_yy, algorithm
            ):
                std_x = np.sqrt(c_xx)
                std_y = np.sqrt(c_yy)
                pearson = c_xy / (std_x * std_y) if c_xy > 0.0 else 0.0
                horizontal = np.sqrt(1.0 + pearson)
                vertical = np.sqrt(1.0 - pearson)
                ellipse = Ellipse(
                    (0, 0),
                    horizontal,
                    vertical,
                    edgecolor=color[alg],
                    facecolor="none",
                    linestyle="--",
                    alpha=0.5,
                )
                transf = (
                    transforms.Affine2D()
                    .rotate_deg(45)
                    .scale(std_x, std_y)
                    .translate(err, rt)
                )
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)
                ax.autoscale()
                # ax.axis('equal')
        g0.add_legend()
        g0.tight_layout()
        experiment = __args.path.split("/")[-1]
        experiment = experiment.split(".")[0]
        g0.figure.canvas.manager.set_window_title(experiment)
        plt.show()


if __name__ == "__main__":
    visualize_stats()
