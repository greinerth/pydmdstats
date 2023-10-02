""" Visualize the damped oscillation in image space
"""
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from pydmd import BOPDMD
from pydmd import VarProDMD
from util.experiment_utils import OPT_ARGS


def generate_complex2d(std: float = -1) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Generate damped oscillating signal

    Args:
        std (float, optional): Standard deviatopm for noise.
                               If <= 0 no noise is added. Defaults to -1.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: snapshots, timestamps, data
    """
    timestamps = np.linspace(0, 6, 16)
    x_1 = np.linspace(-3, 3, 128)
    x_2 = np.linspace(-3, 3, 128)
    x1grid, x2grid = np.meshgrid(x_1, x_2)

    data = [np.expand_dims(2 / np.cosh(x1grid) / np.cosh(x2grid)
                           * (1.2j**-t), axis=0) for t in timestamps]
    snapshots_flat = np.zeros(
        (np.prod(data[0].shape), len(data)), dtype=complex)
    for j, img in enumerate(data):
        __img = img.copy()
        if std > 0:
            __img += np.random.normal(0, std, data[j].shape)
            data[j] = __img
        snapshots_flat[:, j] = np.ravel(__img)
    return snapshots_flat, timestamps, np.concatenate(data, axis=0)


if __name__ == "__main__":

    N_SAMPLES = 4
    CMAP = "plasma"
    STD = 4e-2
    snapshots, time,  data_in = generate_complex2d(STD)
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
    fig1.suptitle('Real Part', fontsize=16)
    fig2.suptitle('Imaginary Part', fontsize=16)

    for i in range(1, N_SAMPLES):
        __varprodmd_img = varprodmd_pred[:, i].reshape(data_in[0].shape)
        __bopdmd_img = bopdmd_pred[:, i].reshape(data_in[0].shape)
        ax1[0][i].imshow(datasub[i].real, cmap="plasma", vmin=-1, vmax=1)
        ax1[1][i].imshow(__bopdmd_img.real, cmap="plasma", vmin=-1, vmax=1)
        ax1[2][i].imshow(__varprodmd_img.real, cmap="plasma", vmin=-1, vmax=1)

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

        ax2[0][i].imshow(datasub[i].imag, cmap=CMAP, vmin=-1, vmax=1)
        ax2[1][i].imshow(__bopdmd_img.imag, cmap=CMAP, vmin=-1, vmax=1)
        ax2[2][i].imshow(__varprodmd_img.imag, cmap=CMAP, vmin=-1, vmax=1)

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

    __optdmd_img = varprodmd_pred[:, 0].reshape(data_in[0].shape)
    __bopdmd_img = bopdmd_pred[:, 0].reshape(data_in[0].shape)

    ax1[0][0].imshow(datasub[0].real, cmap=CMAP, vmin=-1, vmax=1)
    ax1[1][0].imshow(__bopdmd_img.real, cmap=CMAP, vmin=-1, vmax=1)
    ax1[2][0].imshow(__varprodmd_img.real, cmap=CMAP, vmin=-1, vmax=1)

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

    ax2[0][0].imshow(datasub[0].imag, cmap=CMAP, vmin=-1, vmax=1)
    ax2[1][0].imshow(__bopdmd_img.imag, cmap=CMAP, vmin=-1, vmax=1)
    ax2[2][0].imshow(__varprodmd_img.imag, cmap=CMAP, vmin=-1, vmax=1)

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
