#!/usr/bin/env python3
"""Modules for the simulation diagnostics"""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pyevtk
import torch
from torch.utils.tensorboard.writer import SummaryWriter

from pyapes.core.mesh import Mesh


PATHLIKE = Union[str, Path, os.PathLike]


@dataclass
class FieldData:

    mesh: Mesh
    save_dir: PATHLIKE

    def __post_init__(self):

        # Get coordinate origin
        self.g_min = (
            self.mesh.x[0].min() - 0.5 * self.mesh.dx[0],
            self.mesh.x[1].min() - 0.5 * self.mesh.dx[1],
            self.mesh.x[2].min() - 0.5 * self.mesh.dx[2],
        )

        # Get grid spacing
        self.g_dx = (self.mesh.dx[0], self.mesh.dx[1], self.mesh.dx[2])

        # Check log dir and if it doesn't exist, create one
        _validate_log_dir(self.save_dir)


class SaveDictionary(FieldData):
    """Save dictionary data.

    Note:
        - You should provide cell center data not node data.
    """

    def save_data(self, dict_data: dict, file_name: str) -> None:
        """Save dictionary data to a file.

        Args:
            dict_data: data to be saved
            file_name: file name to be saved

        """

        data_save_dict = dict()

        for key in dict_data:

            data_to_save = dict_data[key]

            # If the data is torch.Tensor, convert to numpy
            if type(data_to_save) == torch.Tensor:
                data_to_save = data_to_save.cpu().detach().numpy()

            # Since all meshgrid uses order: "ij" and leading index starts with
            # z, we need to transpose the data
            data_to_save = data_to_save.T

            # If data dtype is booleon, convert to int
            if data_to_save.dtype == np.bool8:
                data_to_save = data_to_save.astype(np.int64)

            data_save_dict.setdefault(key, data_to_save)

        _save_to_vtk(
            self.save_dir, file_name, self.g_min, self.g_dx, data_save_dict
        )


def _save_to_vtk(
    log_dir: PATHLIKE,
    file_name: str,
    g_min: tuple,
    g_dx: tuple,
    save_dict: dict,
):
    """Save dictionary to VTK format.

    Args:
        log_dir (str): location to save
        file_name (str): save file name without extension
        g_min (tuple): grid origin
        g_dx (tuple): grid spacing
        save_dict (dict): dictionary data to save
    """

    log_dir = Path(log_dir, file_name)

    pyevtk.hl.imageToVTK(
        path=str(log_dir),
        origin=g_min,
        spacing=g_dx,
        cellData=save_dict,
    )


class TensorBoardTracker:
    def __init__(self, log_dir: PATHLIKE, overwrite: bool = True):

        self.log_dir = log_dir

        _validate_log_dir(self.log_dir)

        if overwrite:
            pass
        else:
            self.clear_data()

        self._writer = SummaryWriter(log_dir)

    def add_scalar(self, msg: str, value: float, step: int) -> None:
        """Add metric to tensorboard."""

        self._writer.add_scalar(msg, value, step)

    # Not sure yet about this
    def add_figure(self, msg: str, fig: plt.Figure, step: int) -> None:
        """Add matplotlib figure to tensorboard."""

        self._writer.add_figure(msg, fig, step)

    def clear_data(self) -> None:
        """Clear generated data for the tensorboard."""

        # Remove files in the log directory
        for p in Path(self.log_dir).glob("events.*"):
            p.unlink()

        # Remove directory
        Path.rmdir(Path(self.log_dir))

    def flush(self):
        """Flush the data."""

        self._writer.flush()


def _validate_log_dir(log_dir: PATHLIKE) -> None:
    """Check directory and if doesn't exist, create directory."""

    log_dir = Path(log_dir).resolve()

    if log_dir.exists():
        return
    else:
        log_dir.mkdir(parents=True)
