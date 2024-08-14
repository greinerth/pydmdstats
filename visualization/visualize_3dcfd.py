""" Visualize 3D Navier-Stokes reconstructed by VarProDMD"""
from __future__ import annotations

import inspect
import logging
from pathlib import Path

import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
from pydmd import BOPDMD, VarProDMD

logging.basicConfig(level=logging.INFO, filename=__name__)
logging.root.setLevel(logging.INFO)


def densityplot3d(
    ax: plt.axis,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    decay: float = 2.0,
    opacity: float = 1.0,
    cmap: mpl.colors.LinearSegmentedColormap = plt.cm.jet,
    vmin: float | None = None,
    vmax: float | None = None,
    **kwargs,
) -> None:
    """Create a density plot for X, Y, Z coordinates and corresponding intensity values.
    Found on https://github.com/Onkeljoe1234/density_plots_matplotlib/blob/main/Density_plot.ipynb
    Slightly adapted to specific needs.

    :param ax: The axis object to plot the density plot on.
    :type ax: plt.axis
    :param x: Array of X-coordinates
    :type x: np.ndarray
    :param y: Array of y-coordinates
    :type y: np.ndarray
    :param z: Array of z-coordinates
    :type z: np.ndarray
    :param values: Array of values
    :type values: np.ndarray
    :param decay: The decay factor for the alpha values, defaults to 2.0
    :type decay: float, optional
    :param opacity: Opacity value for the alpha value, defaults to 1.0
    :type opacity: float, optional
    :param cmap: Colormap used for mapping intensity values to RGB colors defaults to plt.cm.jet
    :type cmap: mpl.colors.LinearSegmentedColormap, optional
    :param vmin: Minimum value. If None vmin is calculated from `values`, defaults to None
    :type vmin: float | None, optional
    :param vmax: Maximum value. If None vmax is calculated from `values`, defaults to None
    :type vmax: float | None, optional
    """
    # Calculate RGB colors from intensities
    # Normalize the intensities between 0 and 1 and convert them to RGB colors using the chosen colormap
    # for values to be positive
    if vmin is None:
        vmin = np.min(values)

    if vmax is None:
        vmax = np.max(values)

    diff = vmax - vmin

    values = values - vmin
    normed_values = values / diff
    colors = cmap(normed_values)

    # Create alpha values for each data point based on its intensity and the specified decay factor
    alphas = (values / np.max(values)) ** decay
    alphas *= opacity
    colors[:, :, :, 3] = alphas  # add alpha values to RGB values

    # Flatten color array but keep last dimension
    colors_flattened = colors.reshape(-1, colors.shape[-1])

    # Plot a 3D scatter with adjusted alphas
    ax.scatter(x, y, z, c=colors_flattened, **kwargs)


if __name__ == "__main__":
    currentdir = Path(inspect.getfile(inspect.currentframe())).resolve().parent
    file = (
        currentdir.parent
        / "experiments"
        / "data"
        / "3D"
        / "Train"
        / "3D_CFD_Turb_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
    )
    h5file = h5.File(str(file))
    data = h5file["density"][10]
    msg = f"Data shape : {data.shape}"
    logging.info(msg)

    x_coords = np.array(h5file["x-coordinate"][:])
    y_coords = np.array(h5file["y-coordinate"][:])
    z_coords = np.array(h5file["z-coordinate"][:])
    time = np.array(h5file["t-coordinate"][:-1])

    # np.meshgrid(x_coords, y_coords, z_coords)
    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)
    dataflat = np.zeros((np.prod(data.shape[1:]), data.shape[0]))

    for i in range(data.shape[0]):
        dataflat[:, i] = np.ravel(data[i, ...], "F")

    vardmd = VarProDMD(
        optargs={
            "method": "trf",
            "tr_solver": "exact",
            "loss": "linear",
            "x_scale": "jac",
        },
        sorted_eigs="auto",
        # svd_rank=6
    )

    bopdmd = BOPDMD(eig_sort="auto", trial_size=data.shape[-1])

    vardmd.fit(dataflat, time)
    bopdmd.fit(dataflat, time)

    msg = f"VarProDMD omega:\n{vardmd.eigs}"
    logging.info(msg)
    msg = f"BOPDMD omega:\n{bopdmd.eigs}"
    logging.info(msg)

    mrse_vardmd = np.linalg.norm(dataflat - vardmd.forecast(time).real, axis=0).mean()

    mrse_bopdmd = np.linalg.norm(dataflat - bopdmd.forecast(time).real, axis=0).mean()

    msg = f"VarProDMD-RMSE: {mrse_vardmd:.4f}"
    logging.info(msg)
    msg = f"BOPDMD-RMSE: {mrse_bopdmd:.4f}"
    logging.info(msg)

    n_rows = int(np.floor(vardmd.modes.shape[-1] / np.sqrt(vardmd.modes.shape[-1])))
    n_cols = int(np.ceil(vardmd.modes.shape[-1] / n_rows))

    # Visualize VarProDMD-Modes
    figvar, axvar = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "3d"})
    axvar_flat = np.ravel(axvar)

    for i in range(vardmd.modes.shape[-1]):
        mode = np.reshape(vardmd.modes[:, i].real, data.shape[1:], "F")

        densityplot3d(
            axvar_flat[i],
            x_coords[::2, ::2, ::2],
            y_coords[::2, ::2, ::2],
            z_coords[::2, ::2, ::2],
            mode[::2, ::2, ::2],
            cmap=plt.cm.magma,
            marker="s",
            decay=2.0,
        )
        axvar_flat[i].set_title(r"$\boldsymbol{\Phi}_{" + rf"{i + 1}" + r"}$")

    figvar.suptitle("VarProDMD")

    n_rows = int(np.floor(bopdmd.modes.shape[-1] / np.sqrt(bopdmd.modes.shape[-1])))
    n_cols = int(np.ceil(bopdmd.modes.shape[-1] / n_rows))

    # Visualize BOPDMD-Modes
    figbop, axbop = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "3d"})
    axbop_flat = np.ravel(axbop)

    for i in range(bopdmd.modes.shape[-1]):
        mode = np.reshape(bopdmd.modes[:, i].real, data.shape[1:], "F")

        densityplot3d(
            axbop_flat[i],
            x_coords[::2, ::2, ::2],
            y_coords[::2, ::2, ::2],
            z_coords[::2, ::2, ::2],
            mode[::2, ::2, ::2],
            cmap=plt.cm.magma,
            marker="s",
            decay=2.0,
        )
        axbop_flat[i].set_title(r"$\boldsymbol{\Phi}_{" + rf"{i + 1}" + r"}$")

    figbop.suptitle("BOPDMD")

    figeigs, axeigs = plt.subplots(1, 1)

    for i in range(bopdmd.eigs.shape[-1]):
        axeigs.scatter(bopdmd.eigs[i].real, bopdmd.eigs[i].imag, color="r", marker="+")

    axeigs.grid()

    for j in range(i + 1, axvar_flat.shape[-1]):
        figvar.delaxes(axvar_flat[j])

    for i in range(vardmd.eigs.shape[-1]):
        axeigs.scatter(vardmd.eigs[i].real, vardmd.eigs[i].imag, color="b", marker="x")

    for j in range(i + 1, axbop_flat.shape[-1]):
        figbop.delaxes(axbop_flat[j])

    plt.show()
