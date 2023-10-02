""" Visualize the moving points example.
    Two points in image space moving with different velocities.
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydmd import BOPDMD
from pydmd import VarProDMD
from util.experiment_utils import OPT_ARGS, signal2d

def generate_moving_points(std: float = -1) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate moving points example

    Args:
        std (float, optional): Standard deviation, ignored when negative. Defaults to -1.

    Returns:
       Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: snapshots, timestamps, data
    """
    fps = 30.
    total_time = 5.
    velocity = fps / total_time / 4
    x_0 = np.array([-18, -20])
    y_0 = np.array([20, -9])
    timestamps = np.linspace(0, 5, 128)
    __x, __y = np.meshgrid(np.arange(-64, 64), np.arange(-64, 64))
    imgs = np.zeros((timestamps.size, 128, 128))
    snapshots_flat = np.zeros((np.prod(imgs.shape[1:]), timestamps.size))
    for j in range(timestamps.size):
        imgs[j] = signal2d(__x, __y, x_0, y_0, velocity, timestamps[j])
        if std > 0:
            imgs[j] += np.random.normal(0, std, imgs[0].shape)
            snapshots_flat[:, j] = np.ravel(imgs[j])
    return snapshots_flat.astype(np.complex128), timestamps, imgs


if __name__ == "__main__":

    N_SAMPLES = 4
    CMAP = "jet"
    STD = 4e-2
    snapshots, time,  data_in = generate_moving_points(STD)
    sample_dist = data_in.shape[0] // N_SAMPLES
    varprodmd = VarProDMD(optargs=OPT_ARGS)
    varprodmd.fit(snapshots, time)
    bopdmd = BOPDMD()
    bopdmd.fit(snapshots, time)
    varprodmd_pred = varprodmd.forecast(time[::sample_dist])
    bopdmd_pred = bopdmd.forecast(time[::sample_dist])
    datasub = data_in[::sample_dist]

    fig1, ax1 = plt.subplots(3, N_SAMPLES)
    fig2, ax2 = plt.subplots(3, N_SAMPLES)
    # fig1.suptitle('Real Part', fontsize=16)
    fig2.suptitle('Imaginary Part', fontsize=16)

    for i in range(1, N_SAMPLES):
        __varprodmd_img = varprodmd_pred[:, i].reshape(data_in[0].shape)
        __bopdmd_img = bopdmd_pred[:, i].reshape(data_in[0].shape)
        ax1[0][i].imshow(datasub[i].real, cmap=CMAP)
        ax1[1][i].imshow(__bopdmd_img.real, cmap=CMAP)
        ax1[2][i].imshow(__varprodmd_img.real, cmap=CMAP)

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

        ax2[0][i].imshow(datasub[i].imag, cmap=CMAP)
        ax2[1][i].imshow(__bopdmd_img.imag, cmap=CMAP)
        ax2[2][i].imshow(__varprodmd_img.imag, cmap=CMAP)

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

    __varprodmd_img = varprodmd_pred[:, 0].reshape(data_in[0].shape)
    __bopdmd_img = bopdmd_pred[:, 0].reshape(data_in[0].shape)

    ax1[0][0].imshow(datasub[0].real, cmap=CMAP)
    ax1[1][0].imshow(__bopdmd_img.real, cmap=CMAP)
    ax1[2][0].imshow(__varprodmd_img.real, cmap=CMAP)

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

    ax2[0][0].imshow(datasub[0].imag, cmap=CMAP)
    ax2[1][0].imshow(__bopdmd_img.imag, cmap=CMAP)
    ax2[2][0].imshow(__varprodmd_img.imag, cmap=CMAP)

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
