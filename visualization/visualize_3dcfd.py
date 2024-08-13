""" Visualize 3D Navier-Stokes reconstructed by VarProDMD"""
from __future__ import annotations

import inspect
from itertools import combinations
from pathlib import Path

import h5py as h5
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pydmd import VarProDMD

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

    x_coords = h5file["x-coordinate"]
    y_coords = h5file["y-coordinate"]
    z_coords = h5file["z-coordinate"]

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
    fig = make_subplots(rows=n_rows, cols=n_cols, specs=specs)

    for i, j in combinations(range(n_rows), (n_cols)):
        mode_nr = i * n_cols + j
        fig.add_trace(
            go.Volume(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                value=dmd.modes[:, mode_nr].real,
                opacity=0.1,
            ),
            row=i + 1,
            col=j + 1,
        )

    fig.show()
