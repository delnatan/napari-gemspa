"""
This module is an example of a barebones writer plugin for napari.

It implements the Writer specification.
see: https://napari.org/stable/plugins/guides.html?#writers

Replace code below according to your needs.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    DataType = Union[Any, Sequence[Any]]
    FullLayerData = Tuple[DataType, dict, str]


def _save_data(df, path):
    ext = os.path.splitext(path)[1]
    if ext == ".csv":
        sep = ","
    elif ext == ".txt" or ext == ".tsv":
        sep = "\t"
    else:
        raise ValueError(f"{path}: file extension is invalid")

    df.to_csv(path, sep=sep, index=False)


def write_points(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single points layer"""

    # create the pandas data frame of the points data
    # we assume a specific order to axes that is expected in GEMspa points data:
    # if dimension is 2: [y, x]
    # if dimension is 3: [t, y, x]
    # if dimension is 4: [t, z, y, x]

    if data.shape[1] < 2:
        raise ValueError("Points layer data cannot be saved with GEMspa.")

    df = pd.DataFrame()
    if data.shape[1] == 2:
        # df['frame'] = np.zeros(data.shape[0], dtype=int)
        i = 0
    elif data.shape[1] == 3:
        df["frame"] = data[:, 0]
        i = 1
    else:  # data.shape[1] >= 4
        df["frame"] = data[:, 0]
        df["z"] = data[:, 1]
        i = 2
    df["y"] = data[:, i]
    df["x"] = data[:, i + 1]
    for col in meta["properties"].keys():
        df[col] = meta["properties"][col]

    _save_data(df, path)

    # return path to any file(s) that were successfully written
    return [path]


def write_tracks(path: str, data: Any, meta: dict) -> List[str]:
    """Writes a single tracks layer"""

    # create the pandas data frame of the tracks data
    # we assume a specific order to axes that is expected in GEMspa tracks data:
    # if dimension is 4: [track_id, t, z, y, x]
    # if dimension is 3: [track_id, t, y, x]
    if data.shape[1] < 4:
        raise ValueError("Tracks layer data cannot be saved with GEMspa.")

    df = pd.DataFrame()
    df["track_id"] = data[:, 0]
    df["frame"] = data[:, 1]
    if data.shape[1] == 4:
        i = 2
    else:  # data.shape[1] >= 5
        df["z"] = data[:, 2]
        i = 3

    df["y"] = data[:, i]
    df["x"] = data[:, i + 1]
    for col in meta["properties"].keys():
        df[col] = meta["properties"][col]

    # If columns for msd/step-size/radius_gyration, set first frame to empty instead of 0
    # If columns for MSD_fitted and fitting data, set fitting data to empty where MSD_fitted==0

    for col in ["msd", "step_size", "radius_gyration"]:
        if col in df.columns:
            df[col].where(df["frame"] > df["frame_start"], other=np.nan, inplace=True)

    fitting_columns = ["D", "E", "r_sq (lin)", "K", "a", "r_sq (log)"]

    if set(fitting_columns).issubset(set(df.columns)) and "MSD_fitted" in df.columns:
        df[fitting_columns] = df[fitting_columns].where(
            df["MSD_fitted"] == 1, other=np.nan
        )

    _save_data(df, path)

    # return path to any file(s) that were successfully written
    return [path]


# def write_multiple(path: str, data: List[FullLayerData]) -> List[str]:
#     """Writes multiple layers of different types."""
#
#     # implement your writer logic here ...
#
#     # return path to any file(s) that were successfully written
#     return [path]
