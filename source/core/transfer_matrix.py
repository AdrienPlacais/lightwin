#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hold the transfer matrix along the linac.

.. todo::
    Check if it can be more efficient. Maybe store R_xx, R_yy, R_zz separately?

"""
import logging
import numpy as np


class TransferMatrix:
    """
    Hold the (n, 6, 6) transfer matrix along the linac.

    .. note::
        When the simulation is in 1D only, the values corresponding to the
        transverse planes are filled with np.NaN.

    """

    def __init__(self,
                 individual: np.ndarray | list[np.ndarray],
                 is_3d: bool | None = None) -> None:
        """Create the object and compute the cumulated transfer matrix.

        Parameters
        ----------
        individual : np.ndarray | list[np.ndarray]
            Individual transfer matrices.
        is_3d : bool | None, optional
            To indicate if the transfer matrices correspond to a full 3D
            simulation, or only to a longitudinal simulation. The default is
            None, in which case the value of this flag is inferred from the
            dimension of ``individual``.

        """
        if isinstance(individual, list):
            individual = np.array(individual)

        self.n_points = self.individual.shape[0]

        if is_3d is None:
            is_3d = self._determine_if_is_3d(individual)
        self.is_3d = is_3d

        self.individual = individual
        self.cumulated = self._compute_cumulated(is_3d)

    def _compute_cumulated(self, is_3d: bool) -> np.ndarray:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        is_3d : bool
            If the simulation is in 3D or not. If it is in 1D only, we fill
            everything but the [z-dp/p] array with NaN.

        Returns
        -------
        cumulated : np.ndarray
            Cumulated transfer matrix.

        """
        fill_value = np.NaN
        if is_3d:
            fill_value = 0.
        cumulated = np.full((self.n_points, 6, 6), fill_value)

        cumulated[0] = self.individual[0]
        for i in range(1, self.n_points):
            cumulated[i] = self.individual[i] @ cumulated[i - 1]
        return cumulated

    def _determine_if_is_3d(self, individual: np.ndarray) -> bool:
        shape = individual.shape[1:]
        if shape == (2, 2):
            return False
        if shape == (6, 6):
            return True
        logging.error(f"The individual transfer matrices have shape {shape}, "
                      "while (2, 2) (1D) or (6, 6) (3D) are expected.")
        raise IOError("Wrong dimensions for given transfer matrices.")

    @property
    def r_xx(self) -> np.ndarray:
        """Return the transfer matrix of [x-x'] plane."""
        return self.cumulated[:, :2, :2]

    @r_xx.setter
    def r_xx(self, r_xx: np.ndarray) -> None:
        """Set the transfer matrix of [x-x'] plane."""
        self.cumulated[:, :2, :2] = r_xx

    @property
    def r_yy(self) -> np.ndarray:
        """Return the transfer matrix of [y-y'] plane."""
        return self.cumulated[:, 2:4, 2:4]

    @r_yy.setter
    def r_yy(self, r_yy: np.ndarray) -> None:
        """Set the transfer matrix of [y-y'] plane."""
        self.cumulated[:, 2:4, 2:4] = r_yy

    @property
    def r_zz(self) -> np.ndarray:
        """Return the transfer matrix of [z-dp/p] plane."""
        return self.cumulated[:, 4:, 4:]

    @r_zz.setter
    def r_zz(self, r_zz: np.ndarray) -> None:
        """Set the transfer matrix of [z-dp/p] plane."""
        self.cumulated[:, 4:, 4:] = r_zz
