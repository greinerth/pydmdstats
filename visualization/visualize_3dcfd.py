""" Visualize 3D Navier-Stokes reconstructed by VarProDMD"""
from __future__ import annotations

import inspect
from pathlib import Path

import h5py as h5
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pydmd import VarProDMD


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
    Found on https://medium.com/@lorenz.sparrenberg/how-to-create-pretty-3d-density-plots-in-matplotlib-9c76a2f38e59.
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
        / "3D_CFD_Rand_M1.0_Eta1e-08_Zeta1e-08_periodic_Train.hdf5"
    )
    h5file = h5.File(str(file))
    data = h5file["density"][0]

    x_coords = np.array(h5file["x-coordinate"][:])
    y_coords = np.array(h5file["y-coordinate"][:])
    z_coords = np.array(h5file["z-coordinate"][:])
    # np.meshgrid(x_coords, y_coords, z_coords)
    x_coords, y_coords, z_coords = np.meshgrid(x_coords, y_coords, z_coords)
    dataflat = np.zeros((np.prod(data.shape[1:]), data.shape[0]))

    for i in range(data.shape[0]):
        dataflat[:, i] = np.ravel(data[i], "F")

    time = np.linspace(0, 1, dataflat.shape[-1])

    dmd = VarProDMD(
        optargs={"method": "trf", "tr_solver": "exact", "loss": "linear"},
        compression=0.1,
        sorted_eigs=True,
    )

    dmd.fit(dataflat, time)

    n_rows = int(np.floor(np.sqrt(dmd.modes.shape[-1])))
    n_cols = int(np.ceil(dmd.modes.shape[-1] // n_rows))

    specs = (
        np.array([{"type": "volume"}] * (n_rows * n_cols))
        .reshape((n_rows, n_cols))
        .tolist()
    )

    fig, ax = plt.subplots(n_rows, n_cols, subplot_kw={"projection": "3d"})
    ax_flat = np.ravel(ax)

    vmin, vmax = dmd.modes.real.min(), dmd.modes.real.max()
    for i, ax in enumerate(ax_flat):
        mode = np.reshape(dmd.modes[:, i].real, (128, 128, 128), "F")
        mode = mode[::4, ::4, ::4]
        densityplot3d(
            ax,
            x_coords[::4, ::4, ::4],
            y_coords[::4, ::4, ::4],
            z_coords[::4, ::4, ::4],
            mode,
            cmap=plt.cm.magma,
            marker="s",
            decay=5,
        )

    plt.show()
