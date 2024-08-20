from __future__ import annotations

import inspect
import logging
from pathlib import Path

import h5py as h5
import numpy as np
import pyvista as pv
import vtk  # noqa: F401
from varprodmdstatspy import compute_spectral_vorticity_np

logging.basicConfig(level=logging.INFO, filename=__name__)
logging.root.setLevel(logging.INFO)


if __name__ == "__main__":
    pv.set_plot_theme("document")
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
    vx = h5file["Vx"][0]
    vy = h5file["Vy"][0]
    vz = h5file["Vz"][0]
    data = np.concatenate([vx[..., None], vy[..., None], vz[..., None]], axis=-1)
    msg = f"Data shape : {data.shape}"
    logging.info(msg)

    x_coords = np.array(h5file["x-coordinate"][:])
    y_coords = np.array(h5file["y-coordinate"][:])
    z_coords = np.array(h5file["z-coordinate"][:])
    time = np.array(h5file["t-coordinate"][:-1])

    data = data[::4]

    vorts = compute_spectral_vorticity_np(data)
    velocity = np.linalg.norm(vorts, axis=-1)
    # velocity = velocity[::4]

    vmin, vmax = velocity.min(), velocity.max()

    n_rows = int(np.floor(velocity.shape[0] / np.sqrt(velocity.shape[0])))
    n_cols = int(np.ceil(velocity.shape[0] / n_rows))

    plotter = pv.Plotter(shape=(n_rows, n_cols), border=False)
    for i in range(velocity.shape[0]):
        row = i // n_cols
        col = i % n_cols
        plotter.subplot(row, col)
        plotter.add_volume(
            velocity[i],
            cmap="magma",
            clim=(vmin, vmax),
            show_scalar_bar=False,
            opacity="linear",
        )
        time = r"$t_{" + rf"{i * n_cols + 1}" + r"}$"
        plotter.add_text(time, font_size=8, position="upper_edge")

    sargs = {
        "height": 0.75,
        "vertical": True,
        "position_x": 0.8,
        "position_y": 0.05,
        "n_labels": 3,
        "fmt": "%.2f",
    }
    plotter.add_scalar_bar("Velocity [m/s]", **sargs)
    plotter.link_views()
    plotter.window_size = [1200, 531]
    plotter.show()
    # plotter.save_graphic("3dcfd.pdf")
