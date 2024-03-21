"""Visualize the results of BOPDMD vs VarProDMD"""

import argparse
import os
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_stats():
    """Visualize statistics

    Raises:
        FileExistsError: When file does not exist
        ValueError: When experiment is not supported.
    """
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
    if not os.path.exists(__args.path):
        raise FileExistsError(f"{__args.path} does not exist!")

    with open(__args.path, "rb") as handle:
        data = pickle.load(handle)
        df = pd.DataFrame(data)
        print(df)
        # df.rename(columns={"E[t]": r"$E\left[t\right]$ in $s$"})
        if "E[SSIM]" in df.columns:
            df.rename(
                {
                    "E[SSIM]": r"$E\left[\overline{SSIM}\right]$",
                    "E[t]": r"$E\left[t\right]$ in $s$",
                    "STD_NOISE": r"$\sigma_{std}$",
                    "c": r"$c_{comp}$",
                },
                axis="columns",
                inplace=True,
            )
            g0 = sns.FacetGrid(df, col=r"$\sigma_{std}$")
            g0.map_dataframe(
                sns.scatterplot,
                r"$E\left[\overline{SSIM}\right]$",
                r"$E\left[t\right]$ in $s$",
                size="$c_{comp}$",
                legend="full",
                hue="Method",
                alpha=0.5,
            )

        elif "E[RSE]" in df.columns:
            df.rename(
                {
                    "E[RSE]": r"$E\left[d\right]$ in $m$",
                    "E[t]": r"$E\left[t\right]$ in $s$",
                    "STD_NOISE": r"$\sigma_{std}$",
                    "c": r"$c_{comp}$",
                },
                axis="columns",
                inplace=True,
            )
            g0 = sns.FacetGrid(df, col="$\\sigma_{std}$")
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
            raise ValueError("Unsupported Experiment!")

        g0.add_legend()
        plt.show()


if __name__ == "__main__":
    visualize_stats()
