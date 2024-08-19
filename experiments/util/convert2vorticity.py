""" Convert velocity- to vorticity field """
from __future__ import annotations

import argparse
from pathlib import Path

import h5py as h5
import numpy as np
from tqdm import tqdm
from varprodmdstatspy import compute_vorticity


def convert_velocity() -> None:
    parser = argparse.ArgumentParser("Convert velocity field to vorticity!")
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        dest="data",
        help="<Required> Specify path to .hdf5 data file",
    )

    args = parser.parse_args()

    if not Path(args.data).exists():
        msg = f"{args.data} does not exist!"
        raise FileNotFoundError(msg)

    h5file = h5.File(args.data, "r")
    fname = args.data.split("/")[-1]
    fname = fname.split(".hdf5")[0]
    outpath = str(Path(args.data).parent / fname) + "_vorticity.hdf5"

    if not Path(str(outpath)).exists():
        outfile = h5.File(str(outpath), "a")
        outfile.create_dataset(
            "omega_x", shape=h5file["Vx"].shape, dtype=h5file["Vx"].dtype
        )
        outfile.create_dataset(
            "omega_y", shape=h5file["Vy"].shape, dtype=h5file["Vy"].dtype
        )
        outfile.create_dataset(
            "omega_z", shape=h5file["Vz"].shape, dtype=h5file["Vz"].dtype
        )
        outfile.create_dataset(
            "t-coordinate",
            shape=h5file["t-coordinate"].shape,
            dtype=h5file["t-coordinate"].dtype,
        )
        outfile.create_dataset(
            "x-coordinate",
            shape=h5file["x-coordinate"].shape,
            dtype=h5file["x-coordinate"].dtype,
        )
        outfile.create_dataset(
            "y-coordinate",
            shape=h5file["y-coordinate"].shape,
            dtype=h5file["y-coordinate"].dtype,
        )
        outfile.create_dataset(
            "z-coordinate",
            shape=h5file["z-coordinate"].shape,
            dtype=h5file["z-coordinate"].dtype,
        )

        xcoords = h5file["x-coordinate"][:]
        ycoords = h5file["y-coordinate"][:]
        zcoords = h5file["z-coordinate"][:]

        outfile["t-coordinate"][:] = h5file["t-coordinate"][:]
        outfile["x-coordinate"][:] = xcoords
        outfile["y-coordinate"][:] = ycoords
        outfile["z-coordinate"][:] = zcoords

        trials = h5file["Vx"].shape[0]

        for i in tqdm(range(trials), total=trials):
            vx = h5file["Vx"][i][:]
            vy = h5file["Vy"][i][:]
            vz = h5file["Vz"][i][:]
            velocity = np.concatenate(
                [vx[..., None], vy[..., None], vz[..., None]], axis=-1
            )

            vorticity = np.concatenate(
                [
                    compute_vorticity(
                        velocity[j], xcoords, ycoords, zcoords, kind="spectral"
                    )[None]
                    for j in range(velocity.shape[0])
                ],
                axis=0,
            )

            outfile["omega_x"][i] = vorticity[..., 0]
            outfile["omega_y"][i] = vorticity[..., 1]
            outfile["omega_z"][i] = vorticity[..., 2]


if __name__ == "__main__":
    convert_velocity()
