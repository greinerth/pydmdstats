"""Visualize the results of BOPDMD vs VarProDMD
"""
import argparse
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def visualize_stats():
    """Visualize statistics

    Raises:
        FileExistsError: When file does not exist
        ValueError: When experiment is not supported.
    """
    plt.rcParams['text.usetex'] = True
    sns.set_style("whitegrid")

    parser = argparse.ArgumentParser("Visualize Statistics")
    parser.add_argument("-p", "--path", dest="path", type=str,
                        required=True, help="Path to .feather file")
    __args = parser.parse_args()
    if not os.path.exists(__args.path):
        raise FileExistsError(f"{__args.path} does not exist!")

    with open(__args.path, 'rb') as handle:
        data = pickle.load(handle)
        df = pd.DataFrame(data)
        runtimes = np.array(df.loc[:, "E[t]"].to_list())
        compression = np.array(df.loc[:, "c"].to_list())

        noise_levels = np.array(sorted(set(df.loc[:, "STD_NOISE"].to_list())))
        msk = df.loc[:, "Method"].to_numpy() == "BOPDMD"
        idx = np.arange(msk.size)
        bopdmd_idx = idx[msk]
        varpro_idx = idx[~msk]

        for noise in noise_levels:
            __varpro_idx = np.where(
                df.loc[varpro_idx, "STD_NOISE"].to_numpy() == noise)[0]
            __varpro_idx = varpro_idx[__varpro_idx]
            __bopdmd_idx = np.where(
                df.loc[bopdmd_idx, "STD_NOISE"].to_numpy() == noise)[0]
            __bopdmd_idx = bopdmd_idx[__bopdmd_idx]
            __compressions = compression[__varpro_idx]

            varpro_runtime_idx = np.argsort(runtimes[__varpro_idx])
            bopdmd_runtime_idx = np.argsort(runtimes[__bopdmd_idx])

            fastest_varpro = varpro_runtime_idx[0]
            slowest_varpro = varpro_runtime_idx[-1]
            fastest_bopdmd = bopdmd_runtime_idx[0]
            slowest_bopdmd = bopdmd_runtime_idx[-1]

            min_gain = runtimes[__bopdmd_idx[fastest_bopdmd]
                                ] / runtimes[__varpro_idx[slowest_varpro]]
            max_gain = runtimes[__bopdmd_idx[slowest_bopdmd]
                                ] / runtimes[__varpro_idx[fastest_varpro]]

            print(f"\nNoise level: {noise}")
            min_gain_str = "Minimum gain: {:.4f}, VarPro Compression: {}"
            max_gain_str = "Maximum gain: {:.4f}, VarPro Compression: {}"
            min_gain_str = min_gain_str.format(min_gain, __compressions[slowest_varpro])
            max_gain_str = max_gain_str.format(max_gain, __compressions[fastest_varpro])
            str_len = max(len(min_gain_str), len(max_gain_str))

            print(str_len * "=")
            print(min_gain_str)
            print(max_gain_str)

        print("\n")
        print(df)

        # df.rename(columns={"E[t]": r"$E\left[t\right]$ in $s$"})
        if "E[SSIM]" in df.columns:
            df.rename({"E[SSIM]": r"$E\left[\overline{SSIM}\right]$",
                       "E[t]": r"$E\left[t\right]$ in $s$",
                       "STD_NOISE": r"$\sigma_{std}$",
                       "c": r"$c_{comp}$"},
                      axis="columns",
                      inplace=True)
            g0 = sns.FacetGrid(df, col=r"$\sigma_{std}$")
            g0.map_dataframe(sns.scatterplot,
                             r"$E\left[\overline{SSIM}\right]$",
                             r"$E\left[t\right]$ in $s$",
                             size="$c_{comp}$",
                             legend="full",
                             hue="Method",
                             alpha=0.5)

        elif "E[RSE]" in df.columns:
            df.rename({"E[RSE]": r"$E\left[d\right]$ in $m$",
                       "E[t]": r"$E\left[t\right]$ in $s$",
                       "STD_NOISE": r"$\sigma_{std}$",
                       "c": r"$c_{comp}$"},
                      axis="columns",
                      inplace=True)
            g0 = sns.FacetGrid(df, col="$\\sigma_{std}$")
            g0.map_dataframe(sns.scatterplot,
                             r"$E\left[d\right]$ in $m$",
                             r"$E\left[t\right]$ in $s$",
                             size="$c_{comp}$",
                             legend="full",
                             hue="Method",
                             alpha=0.5)

        else:
            raise ValueError("Unsupported Experiment!")

        g0.add_legend()
        plt.show()


if __name__ == "__main__":
    visualize_stats()
