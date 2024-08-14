"""
Adapted pdebench data-download taken from
https://github.com/pdebench/PDEBench/blob/main/pdebench/data_download/download_direct.py.
The module avoids pytorch for data download.
"""
from __future__ import annotations

import argparse
import inspect
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, filename=__name__)
logging.root.setLevel(logging.INFO)


def parse_metadata(pde_names: list[str]) -> pd.DataFrame:
    """Parse metadata given list of PDEs

    Options for pde_names:
    - Advection
    - Burgers
    - 1D_CFD
    - Diff-Sorp
    - 1D_ReacDiff
    - 2D_CFD
    - Darcy
    - 2D_ReacDiff
    - NS_Incom
    - SWE
    - 3D_CFD

    :param pde_names: List of Pdes
    :type pde_names: list[str]
    :return: Filtered dataframe containing metadata of files to be downloaded
    :rtype: pd.DataFrame
    """
    path = Path(inspect.getfile(inspect.currentframe())).resolve().parent
    meta_df = pd.read_csv(path / "pdebench_data_urls.csv")

    # Ensure the pde_name is defined
    pde_list = [
        "advection",
        "burgers",
        "1d_cfd",
        "diff_sorp",
        "1d_reacdiff",
        "2d_cfd",
        "darcy",
        "2d_reacdiff",
        "ns_incom",
        "swe",
        "3d_cfd",
    ]

    assert all(name.lower() in pde_list for name in pde_names), "PDE name not defined."

    # Filter the files to be downloaded
    meta_df["PDE"] = meta_df["PDE"].str.lower()
    return meta_df[meta_df["PDE"].isin(pde_names)]


def download_data(root_folder: str, pde_name: list[str]) -> None:
    """Download data splits specific to a given PDE.

    :param root_folder: The root folder where the data will be downloaded
    :type root_folder: str
    :param pde_name: The name of the PDE for which the data to be downloaded
    :type pde_name: list[str]
    """
    msg = f"Downloading data for {pde_name} ..."
    logging.info(msg)

    # Load and parse metadata csv file
    pde_df = parse_metadata(pde_name)

    # Iterate filtered dataframe and download the files
    for _, row in tqdm(pde_df.iterrows(), total=pde_df.shape[0]):
        file_path = Path(root_folder) / row["Path"]
        if not Path(file_path).exists():
            Path(file_path).mkdir(parents=True)

        fname = str(file_path / row["Filename"])
        if not Path(fname).exists():
            proc = subprocess.Popen(
                ["wget", "-O", fname, row["URL"]], stdout=subprocess.PIPE
            )
            for line in proc.stdout:
                sys.stdout.write(line)


def pdebench_downdload() -> None:
    """Download pdebench data"""

    arg_parser = argparse.ArgumentParser(
        prog="Download Script",
        description="Helper script to download the PDEBench datasets",
        epilog="",
    )

    arg_parser.add_argument(
        "--root_folder",
        type=str,
        default=Path(inspect.getfile(inspect.currentframe())).resolve().parent.parent
        / "data",
        help="Root folder where the data will be downloaded",
    )
    arg_parser.add_argument(
        "--pde_name",
        action="append",
        help="Name of the PDE dataset to download. You can use this flag multiple times to download multiple datasets",
    )

    args = arg_parser.parse_args()

    download_data(args.root_folder, args.pde_name)


if __name__ == "__main__":
    pdebench_downdload()
