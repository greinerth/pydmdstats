"""Visualize library selection scheme
"""
import matplotlib.pyplot as plt
import numpy as np

from util.experiment_utils import signal, OPT_ARGS
from pydmd.varprodmd import VarProDMD

if __name__ == "__main__":

    plt.rc('text', usetex=True)

    COMP = 0.8
    time = np.linspace(0, 4 * np.pi, 100)
    x_loc = np.linspace(-10, 10, 1024)
    __x, __time = np.meshgrid(x_loc, time)
    z_signal = signal(__x, __time).T
    __x = __x.T
    __time = __time.T
    __dmd = VarProDMD(compression=COMP, optargs=OPT_ARGS, sorted_eigs=False, exact=False)
    __dmd.fit(z_signal, time)
    print(__dmd.eigs.imag)
    __indices = __dmd.selected_samples
    __pred = __dmd.forecast(time)

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.plot_surface(__time, __x, z_signal.real, cmap="plasma", alpha=0.5)

    if __indices is not None:
        for i in range(__indices.size):
            ax.plot(__time[:, __indices[i]], __x[:, __indices[i]],
                    z_signal[:, __indices[i]].real, "--", color="r", linewidth=1)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Original - $\Re\{\mathbf{X}\}$")

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.plot_surface(__time, __x, z_signal.imag, cmap="plasma", alpha=0.5)
    if __indices is not None:
        for i in range(__indices.size):
            ax.plot(__time[:, __indices[i]], __x[:, __indices[i]],
                    z_signal[:, __indices[i]].imag, "--", color="r", linewidth=1)
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Original - $\Im\{\mathbf{X}\}$")

    fig = plt.figure()
    ax = fig.add_subplot(2, 1, 1, projection='3d')
    ax.plot_surface(__time, __x, __pred.real, cmap="plasma")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Reconstructed - $\Re\{\hat{\mathbf{X}}\}$")

    ax = fig.add_subplot(2, 1, 2, projection='3d')
    ax.plot_surface(__time, __x, __pred.imag, cmap="plasma")
    ax.set_xlabel(r"$s$")
    ax.set_ylabel(r"$m$")
    ax.set_title(r"Reconstructed - $\Im\{\hat{\mathbf{X}}\}$")

    plt.show()
