"""
# XRR Runfile Generator

This module is used to generate a runfile for the XRR experiment at the ALS beamline
"""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import ipywidgets as widgets

DATA_PATH = (
    Path("Washington State University (email.wsu.edu)")
    / "Carbon Lab Research Group - Documents"
    / "Synchrotron Logistics and Data"
    / "ALS - Berkeley"
    / "Data"
    / "BL1101"
)

# Ipywidgets command to construct a fillable form used to update the config.yaml file
# with the necessary information for the XRR experiment



def unique_filename(path: Path) -> Path:
    """Generate a unique filename."""
    i = 1
    while path.exists():
        path = path.with_name(f"{path.stem}({i}){path.suffix}")
        i += 1
    return path


def load_config(config: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the config.yaml file."""
    with config.open("rb") as f:
        config = yaml.safe_load(f)

    df = pd.DataFrame(config["energy"])
    return df, config["config"]


def process_stitch(
    theta_i, theta_f, hos, hes, energy, et, points_per_fringe, fringe_size
) -> pd.DataFrame:
    """Process a stitch."""
    # Use q space instead of theta space
    q_i = np.sin(np.deg2rad(theta_i)) * 4 * np.pi / (12.4 / energy)
    q_f = np.sin(np.deg2rad(theta_f)) * 4 * np.pi / (12.4 / energy)

    n_points = int(np.ceil((q_f - q_i) / fringe_size) * points_per_fringe)

    q = np.linspace(q_i, q_f, n_points)
    theta = np.rad2deg(np.arcsin(q * (12.4 / energy) / (4 * np.pi)))
    ccd = 2 * theta
    hos = np.full_like(theta, hos)
    hes = np.full_like(theta, hes)
    et = np.full_like(theta, et)

    return pd.DataFrame({"theta": theta, "ccd": ccd, "hos": hos, "hes": hes, "et": et})


def add_overlap(
    current_df: pd.DataFrame, last_df: pd.DataFrame, overlap: int, repeat: int
) -> pd.DataFrame:
    """Add overlap to the current_df."""
    # Add back the overlap points
    overlap_df = last_df.iloc[-overlap:]
    overlap_df["hos"] = current_df.iloc[0]["hos"]
    overlap_df["hes"] = current_df.iloc[0]["hes"]
    overlap_df["et"] = current_df.iloc[0]["et"]
    # In the overlap_df repeat the first point repeat times
    overlap_df = pd.concat([overlap_df.iloc[0:1]] * repeat)
    return pd.concat([overlap_df, current_df])


def process_energy(df_slice: pd.DataFrame, config: dict, energy: float) -> pd.DataFrame:
    """Generate a chunk for a single energy."""
    theta = df_slice.loc["theta"]
    hos = df_slice.loc["hos"]
    hes = df_slice.loc["hes"]
    et = df_slice.loc["et"]

    # This is parameterized by the number of points in a fringe
    points_per_fringe = config["collection"]["density"]
    fringe_size = 2 * np.pi / config["collection"]["thickness"]

    theta_pairs = [(theta[i], theta[i + 1]) for i in range(len(theta) - 1)]
    energy_df = []
    for i, (t_i, t_f) in enumerate(theta_pairs):
        stitch_df = process_stitch(
            t_i, t_f, hos[i], hes[i], energy, et[i], points_per_fringe, fringe_size
        )
        stitch_df["x"] = config["geometry"]["x"] + i * 0.1
        energy_df.append(stitch_df)

    energy_df = pd.concat(energy_df)
    energy_df["energy"] = energy
    return energy_df


def generate_runfile(macro_folder=str | Path) -> None:
    """
    Generate a run file.

    Parameters
    ----------
    macro_folder : str | Path
        Location of the macro folder

    Returns
    -------
    None

    Outputs a run file macro for the ALS beamline 11.0.1.2 at the Advanced Light Source.

    Example
    -------
    >>> generate_runfile("sample1", path_to_save)
    File
    >>> Sample X --> ... Sample Theta --> CCD Theta --> ... Beamline Energy -->
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>>   ⋮                ⋮                  ⋮                   ⋮                 ⋮
    >>> 12.4     --> ... 70            --> 140         --> ... 319          --> 10

    Columns
    -------
    Sample X : float
        Sample X position

    Sample Y : float
        Sample Y position

    Sample Z : float
        Sample Z position

    Sample Theta : float
        Sample Theta position

    CCD Theta : float
        CCD Theta position

    Higher Order Suppressor : float
        Higher Order Suppressor position

    Horizontal Exit Slit Size: float
        Horizontal Exit Slit position

    Beamline Energy : float
        Beamline Energy

    Exposure Time : float
        Exposure Time - This column has no label in the run file
    """
    df_stitches, config = load_config(Path.cwd() / "config.yaml")
    save_path = Path(macro_folder) / f"{config["name"]}.txt"
    # Generate a new name if the file allready exists

    df = []
    for i, en in enumerate(df_stitches.columns):
        energy_df = process_energy(df_stitches[en], config, float(en))
        energy_df["y"] = config["geometry"]["y"] + i * 0.1
        df.append(energy_df)

    df = pd.concat(df)
    df["z"] = config["geometry"]["z"]
    print(df)
    df = df.reindex(
        columns=[
            "x",
            "y",
            "z",
            "theta",
            "ccd",
            "hos",
            "hes",
            "energy",
            "et",
        ]
    )
    if save_path.exists():
        # Check if ther are changes in the file
        save_path = unique_filename(save_path)

    df.to_csv(
        save_path,
        sep="\t",
        index=False,
        header=[
            "Sample X",
            "Sample Y",
            "Sample Z",
            "Sample Theta",
            "CCD Theta",
            "Higher Order Suppressor",
            "Horizontal Exit Slit Size",
            "Beamline Energy",
            "",
        ],
    )
    print(df)
    return df, config["name"]

    # Construct the runfile


def runfile():S
    """
    Constructes the macro for the XRR experiment allong with saving the data.

    Example
    -------
    >>> generate_runfile("sample1", path_to_save)
    File
    >>> Sample X --> ... Sample Theta --> CCD Theta --> ... Beamline Energy -->
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>> 12.4     --> ... 0.0           --> 0.0         --> ... 250          --> .001
    >>>   ⋮                ⋮                  ⋮                   ⋮                 ⋮
    >>> 12.4     --> ... 70            --> 140         --> ... 319          --> 10

    Columns
    -------
    Sample X : float
        Sample X position

    Sample Y : float
        Sample Y position

    Sample Z : float
        Sample Z position

    Sample Theta : float
        Sample Theta position

    CCD Theta : float
        CCD Theta position

    Higher Order Suppressor : float
        Higher Order Suppressor position

    Horizontal Exit Slit Size: float
        Horizontal Exit Slit position

    Beamline Energy : float
        Beamline Energy

    Exposure Time : float
        Exposure Time - This column has no label in the run file
    """
    # Create the save location for the data
    date = datetime.datetime.now()
    beamtime = f"{date.strftime('%Y%b')}/XRR/"
    date = date.strftime("%Y %m %d")

    data_path = Path.home() / DATA_PATH
    save_path = data_path / beamtime / ".macro" / date
    all_data_path = data_path / "XRR"

    if not save_path.exists():
        save_path.mkdir(parents=True)
        print(f"Created {save_path}")

    if not all_data_path.exists():
        all_data_path.mkdir(parents=True)
        print(f"Created {all_data_path}")

    # Generate the runfile
    df, name = generate_runfile(save_path)
    df["name"] = name
    df["date-time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if (all_data_path / "all_data.parqet").exists():
        all_data = pd.read_csv(all_data_path / "all_data.parquet")
        all_data = pd.concat([all_data, df])
    else:
        all_data = df

    all_data.to_parquet(all_data_path / "all_data.parquet", index=False)
