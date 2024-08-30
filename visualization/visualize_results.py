"""Visualize the results of BOPDMD vs VarProDMD"""
from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import axis, transforms
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch

logging.basicConfig(level=logging.INFO)
logging.root.setLevel(logging.INFO)


def confidence_ellipse(
    xcoord: float,
    ycoord: float,
    cxx: float,
    cxy: float,
    cyy: float,
    ax: axis,
    n_std: float = 3.0,
    facecolor: str = "none",
    **kwargs,
) -> Patch:
    """Function taken from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
    Draw a 2D confidence ellipse

    :param xcoord: x coordinate of 2D ellipse
    :type xccord: float
    :param ycoord: y coordinate of 2D ellipse
    :type ycoord: float
    :param ax: Axis of (sub)plot
    :type ax: axis
    :param n_std: Set according to confidence of ellipse, defaults to 3.0
    :type n_std: float, optional
    :param facecolor: Set face color, defaults to 'none'
    :type facecolor: str, optional
    :return: 2D Arist with face and color
    :rtype: Patch
    """

    pearson = cxy / np.sqrt(cxx * cyy)

    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs,
    )

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cxx) * n_std

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cyy) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(xcoord, ycoord)
    )

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def visualize_stats():
    """Visualize statistics

    :raises FileExistsError: When file does not exist
    :raises ValueError: When experiment is not supported.
    """

    dpi = 300.0
    height = 531.0
    width = 1200.0

    # plt.style.use("science")
    plt.rcParams["text.usetex"] = True
    plt.rcParams["figure.dpi"] = dpi
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
    parser.add_argument(
        "--logx", dest="logx", action="store_true", help="Logscale for x-axis"
    )
    parser.add_argument(
        "--logy", dest="logy", action="store_true", help="Logscale for y-axis"
    )
    parser.add_argument(
        "--sharex",
        dest="sharex",
        action="store_true",
        help="Shared x-axis for the plots",
    )
    parser.add_argument(
        "--sharey",
        dest="sharey",
        action="store_true",
        help="Shared y-axis for the plots",
    )

    parser.add_argument(
        "--conf",
        type=float,
        dest="conf",
        default=1.0,
        help="Factor for premultiplying the standard deviations (confidence). [Default: 1.0]",
    )

    args = parser.parse_args()
    nstd = abs(args.conf)

    if not Path(args.path).exists():
        msg = f"{args.path} does not exist!"
        raise FileExistsError(msg)

    with Path(args.path).open("rb") as handle:
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
            rowname = r"$E\left[\overline{SSIM}\right]$"
            colname = r"$E\left[t\right]$ in $s$"

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
            rowname = r"$E\left[d\right]$ in $m$"
            colname = r"$E\left[t\right]$ in $s$"

        else:
            msg = "Unsupported Experiment!"
            raise ValueError(msg)

        g0 = sns.FacetGrid(
            dataframe, col=r"$\sigma_{std}$", sharex=args.sharex, sharey=args.sharey
        )
        g0.map_dataframe(
            sns.scatterplot,
            rowname,
            colname,
            size="$c_{comp}$",
            legend="full",
            hue="Method",
            alpha=0.5,
        )
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
                # no dengerate cases, a confidence ellipse can be drawn
                if (
                    not np.isclose(c_xy, 0.0)
                    and not np.isclose(c_xy, 0.0)
                    and not np.isclose(c_yy, 0.0)
                ):
                    confidence_ellipse(
                        err,
                        rt,
                        c_xx,
                        c_xy,
                        c_yy,
                        ax,
                        n_std=nstd,
                        alpha=0.5,
                        linestyle="--",
                        edgecolor=color[alg],
                    )
                # no correspondence between runtime and error, but mainly error is deviating
                elif (
                    np.isclose(c_xy, 0.0)
                    and not np.isclose(c_xx, 0.0)
                    and np.isclose(c_yy, 0.0)
                ):
                    stdx = np.sqrt(c_xx)
                    ax.add_line(
                        Line2D(
                            [err - nstd * stdx, err + nstd * std],
                            [rt, rt],
                            linestyle="--",
                            alpha=0.5,
                            color=color[alg],
                        )
                    )

                # no correspondence between runtime and error, but mainly runtime is deviating
                elif (
                    np.isclose(c_xy, 0.0)
                    and np.isclose(c_xx, 0.0)
                    and not np.isclose(c_yy, 0.0)
                ):
                    stdy = np.sqrt(c_yy)
                    ax.add_line(
                        Line2D(
                            [err, err],
                            [rt - nstd * stdy, rt + nstd * stdy],
                            linestyle="--",
                            alpha=0.5,
                            color=color[alg],
                        )
                    )
                else:
                    msg = "Degenerate case encountered! Cannot draw uncertainty ellipse or lines."
                    logging.warning(msg)

                ax.autoscale()

        g0.add_legend()
        g0.tight_layout()
        experiment = args.path.split("/")[-1]
        experiment = experiment.split(".")[0]
        g0.figure.canvas.manager.set_window_title(experiment)
        g0.figure.set_size_inches(width / dpi, height / dpi)

        if args.logx and args.logy:
            g0.set(xscale="log", yscale="log")
        elif args.logx and not args.logy:
            g0.set(xscale="log")
        elif not args.logx and args.logy:
            g0.set(yscale="log")

        # g0.set(xscale="log")
        plt.show()


if __name__ == "__main__":
    visualize_stats()
