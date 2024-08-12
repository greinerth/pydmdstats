"""Visualize library selection scheme"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pydmd.varprodmd import VarProDMD

from util.experiment_utils import signal

if __name__ == "__main__":
    plt.style.use("science")
    OPT_ARGS = {"method": "trf", "tr_solver": "exact", "loss": "linear"}

    plt.rc("text", usetex=True)

    COMP = 0.8
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    _x, _time = np.meshgrid(x_loc, time)
    z_signal = signal(_x, _time).T
    _x = _x.T
    _time = _time.T

    dmd = VarProDMD(compression=COMP, optargs=OPT_ARGS, exact=True)
    dmd.fit(z_signal, time)

    indices = dmd.selected_samples
    pred = dmd.forecast(time)

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(_time, _x, z_signal.real, cmap="plasma", alpha=0.5)

    if indices is not None:
        for i in range(indices.size):
            ax.plot(
                _time[:, indices[i]],
                _x[:, indices[i]],
                z_signal[:, indices[i]].real,
                "--",
                color="r",
                linewidth=1,
            )
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Original - $\Re\{\mathbf{X}\}$")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(_time, _x, z_signal.imag, cmap="plasma", alpha=0.5)
    if indices is not None:
        for i in range(indices.size):
            ax.plot(
                _time[:, indices[i]],
                _x[:, indices[i]],
                z_signal[:, indices[i]].imag,
                "--",
                color="r",
                linewidth=1,
            )
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Original - $\Im\{\mathbf{X}\}$")

    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(_time, _x, pred.real, cmap="plasma")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Reconstructed - $\Re\{\hat{\mathbf{X}}\}$")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(_time, _x, pred.imag, cmap="plasma")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Reconstructed - $\Im\{\hat{\mathbf{X}}\}$")

    plt.show()
