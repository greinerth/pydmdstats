"""Visualize the global temperature in image space"""

import inspect
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from pydmd import BOPDMD, VarProDMD

# from util.experiment_utils import OPT_ARGS
from varprodmd_ssim_performance import download


def generate_global_temp(
    std: float = -1,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Read Sea Surface Temperature and augment with noise.

    Args:
        std (float, optional): Standard deviatopm for noise.
                               If <= 0 no noise is added. Defaults to -1.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: snapshots, timestamps, data
    """
    DATASET = "sst.day.mean.ltm.1982-2010.nc"
    YEARS = 2010 - 1982
    currentdir = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    FILE = os.path.join(currentdir, "data")

    if not os.path.exists(FILE):
        os.makedirs(FILE)
        download(
            "https://downloads.psl.noaa.gov/Datasets/noaa.oisst.v2.highres/sst.day.mean.ltm.1982-2010.nc",
            os.path.join(FILE, DATASET),
        )

    FILE = os.path.join(FILE, DATASET)
    dataset = nc.Dataset(FILE)
    sst = dataset["sst"][:]

    low, high = dataset["sst"].valid_range
    n_samples = float(sst.shape[0])
    sst = sst[-128:]

    dt = (float(YEARS) / float(n_samples)) * sst.shape[0]

    timestamps = np.linspace(0, dt, sst.shape[0])
    img0 = sst[0]
    # img0 = img0[::-1, ::]
    msk = ~(img0 < low)
    msk &= ~(img0 > high)

    # remember invalid numbers from flipped original image
    # need to invert once again else not the correct values are remembered
    rows, cols = np.where(~msk)
    sst[..., rows, cols] = 0.0
    __msk_flat = np.ravel(msk)
    __flat = np.zeros(shape=(np.sum(msk), sst.shape[0]))
    for j in range(sst.shape[0]):
        __img = sst[j]
        __noisy = __img.copy()
        if std > 0:
            __noisy += np.random.normal(0.0, std, size=__img.shape)
        __flat[:, j] = np.ravel(__noisy)[__msk_flat]
    __flat = __flat.astype(np.complex128)
    return __flat, timestamps, sst, __msk_flat


if __name__ == "__main__":
    OPT_ARGS = {"method": "trf", "tr_solver": "exact", "loss": "linear"}

    N_SAMPLES = 3
    CMAP = "jet"
    STD = 0.01
    snapshots, time, data_in, msk_flat = generate_global_temp(STD)
    sample_dist = data_in.shape[0] // N_SAMPLES
    varprodmd = VarProDMD(optargs=OPT_ARGS, exact=False)
    varprodmd.fit(snapshots, time)

    bopdmd = BOPDMD(trial_size=data_in.shape[0])
    bopdmd.fit(snapshots, time)
    varprodmd_pred = varprodmd.forecast(time[::sample_dist])
    bopdmd_pred = bopdmd.forecast(time[::sample_dist])
    datasub = data_in[::sample_dist]

    fig1, ax1 = plt.subplots(3, N_SAMPLES)
    fig2, ax2 = plt.subplots(3, N_SAMPLES)
    # fig1.suptitle('Real Part', fontsize=16)
    fig2.suptitle("Imaginary Part", fontsize=16)

    for i in range(1, N_SAMPLES):
        __varprodmd_img_flat = np.zeros(
            (
                np.prod(
                    data_in.shape[1:],
                )
            ),
            dtype=varprodmd_pred.dtype,
        )
        __varprodmd_img_flat[msk_flat] = varprodmd_pred[:, i]
        __bopdmd_img_flat = np.zeros_like(__varprodmd_img_flat)
        __bopdmd_img_flat[msk_flat] = bopdmd_pred[:, i]
        __varprodmd_img = __varprodmd_img_flat.reshape(data_in[0].shape)
        __bopdmd_img = __bopdmd_img_flat.reshape(data_in[0].shape)
        ax1[0][i].imshow(datasub[i][::-1, :].real, cmap=CMAP)
        ax1[1][i].imshow(__bopdmd_img[::-1, :].real, cmap=CMAP)
        ax1[2][i].imshow(__varprodmd_img[::-1, :].real, cmap=CMAP)

        ax1[0][i].xaxis.set_tick_params(labelbottom=False)
        ax1[0][i].yaxis.set_tick_params(labelleft=False)
        ax1[1][i].xaxis.set_tick_params(labelbottom=False)
        ax1[1][i].yaxis.set_tick_params(labelleft=False)
        ax1[2][i].xaxis.set_tick_params(labelbottom=False)
        ax1[2][i].yaxis.set_tick_params(labelleft=False)

        ax1[0][i].set_xticks([])
        ax1[1][i].set_xticks([])
        ax1[2][i].set_xticks([])

        ax1[0][i].set_yticks([])
        ax1[1][i].set_yticks([])
        ax1[2][i].set_yticks([])

        ax2[0][i].imshow(datasub[i][::-1, :].imag, cmap=CMAP)
        ax2[1][i].imshow(__bopdmd_img[::-1, :].imag, cmap=CMAP)
        ax2[2][i].imshow(__varprodmd_img[::-1, :].imag, cmap=CMAP)

        ax2[0][i].xaxis.set_tick_params(labelbottom=False)
        ax2[0][i].yaxis.set_tick_params(labelleft=False)
        ax2[1][i].xaxis.set_tick_params(labelbottom=False)
        ax2[1][i].yaxis.set_tick_params(labelleft=False)
        ax2[2][i].xaxis.set_tick_params(labelbottom=False)
        ax2[2][i].yaxis.set_tick_params(labelleft=False)

        ax2[0][i].set_xticks([])
        ax2[1][i].set_xticks([])
        ax2[2][i].set_xticks([])

        ax2[0][i].set_yticks([])
        ax2[1][i].set_yticks([])
        ax2[2][i].set_yticks([])

    __varprodmd_img_flat = np.zeros(
        (np.prod(data_in.shape[1:]),), dtype=varprodmd_pred.dtype
    )
    __varprodmd_img_flat[msk_flat] = varprodmd_pred[:, 0]
    __bopdmd_img_flat = np.zeros_like(__varprodmd_img_flat)
    __bopdmd_img_flat[msk_flat] = bopdmd_pred[:, 0]
    __varprodmd_img = __varprodmd_img_flat.reshape(data_in[0].shape)
    __bopdmd_img = __bopdmd_img_flat.reshape(data_in[0].shape)

    ax1[0][0].imshow(datasub[0][::-1, :].real, cmap=CMAP)
    ax1[1][0].imshow(__bopdmd_img[::-1, :].real, cmap=CMAP)
    ax1[2][0].imshow(__varprodmd_img[::-1, :].real, cmap=CMAP)

    ax1[0][0].set_ylabel("Original", weight="bold")
    ax1[0][0].xaxis.set_tick_params(labelbottom=False)

    ax1[1][0].set_ylabel("BOPDMD", weight="bold")
    ax1[1][0].xaxis.set_tick_params(labelbottom=False)

    ax1[2][0].set_ylabel("VarProDMD", weight="bold")
    ax1[2][0].xaxis.set_tick_params(labelbottom=False)

    ax1[0][0].set_xticks([])
    ax1[1][0].set_xticks([])
    ax1[2][0].set_xticks([])
    ax1[0][0].set_yticks([])
    ax1[1][0].set_yticks([])
    ax1[2][0].set_yticks([])

    ax2[0][0].imshow(datasub[0][::-1, :].imag, cmap=CMAP)
    ax2[1][0].imshow(__bopdmd_img[::-1, :].imag, cmap=CMAP)
    ax2[2][0].imshow(__varprodmd_img[::-1, :].imag, cmap=CMAP)

    ax2[0][0].set_ylabel("Original", weight="bold")
    ax2[0][0].xaxis.set_tick_params(labelbottom=False)

    ax2[1][0].set_ylabel("BOPDMD", weight="bold")
    ax2[1][0].xaxis.set_tick_params(labelbottom=False)

    ax2[2][0].set_ylabel("VarProDMD", weight="bold")
    ax2[2][0].xaxis.set_tick_params(labelbottom=False)

    ax2[0][0].set_xticks([])
    ax2[1][0].set_xticks([])
    ax2[2][0].set_xticks([])
    ax2[0][0].set_yticks([])
    ax2[1][0].set_yticks([])
    ax2[2][0].set_yticks([])

    plt.show()
