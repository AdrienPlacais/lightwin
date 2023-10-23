#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hold the transfer matrix along the linac.

.. todo::
    Check if it can be more efficient. Maybe store R_xx, R_yy, R_zz separately?

.. todo::
    Handle when treating sub-linacs. First cumulated transfer matrix then sould
    not be ``eye`` matrix.

"""
import logging
import numpy as np


class TransferMatrix:
    """
    Hold the (n, 6, 6) transfer matrix along the linac.

    .. note::
        When the simulation is in 1D only, the values corresponding to the
        transverse planes are filled with np.NaN.

    Attributes
    ----------
    individual : np.ndarray
        Individual transfer matrices along the linac. Not defined if not
        provided at initialisation.
    cumulated : np.ndarray
        Cumulated transfer matrices along the linac.

    """

    def __init__(self,
                 individual: np.ndarray | list[np.ndarray] | None = None,
                 cumulated: np.ndarray | None = None,
                 insert_eye_matrix: bool = True,
                 ) -> None:
        """Create the object and compute the cumulated transfer matrix.

        Parameters
        ----------
        individual : np.ndarray | list[np.ndarray] | None, optional
            Individual transfer matrices. The default is None, in which case
            the ``cumulated`` transfer matrix must be provided directly.
        cumulated : np.ndarray | None, optional
            Cumulated transfer matrices. The default is None, in which case the
            ``individual`` transfer matrices must be given.
        insert_eye_matrix : bool, optional
            If an eye matrix should be inserted at the first position of
            ``cumulated``. The default is True.

        """
        if isinstance(individual, list):
            individual = np.array(individual)

        self.individual: np.ndarray
        if individual is not None:
            self.individual = individual
            is_3d, n_points, cumulated = self._init_from_individual(individual)

        else:
            is_3d, n_points, cumulated = self._init_from_cumulated(cumulated)

        self.is_3d = is_3d
        self.n_points = n_points

        if insert_eye_matrix:
            cumulated = self._insert_eye_transfer_matrix_at_start(cumulated)
        self.cumulated = cumulated

    def _init_from_individual(self, individual: np.ndarray
                              ) -> tuple[bool, int, np.ndarray]:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        individual : np.ndarray
            Individual transfer matrices along the linac.

        Returns
        -------
        is_3d : bool
            If the simulation is in 3D or not.
        n_points : int
            Number of mesh points along the linac.
        cumulated : np.ndarray
            Cumulated transfer matrices.

        """
        is_3d = self._determine_if_is_3d(individual)
        n_points = individual.shape[0]
        cumulated = self._compute_cumulated(is_3d, n_points)
        return is_3d, n_points, cumulated

    def _init_from_cumulated(self, cumulated: np.ndarray | None
                             ) -> tuple[bool, int, np.ndarray]:
        """Check that the given cumulated matrix is valid.

        Parameters
        ----------
        cumulated : np.ndarray
            Cumulated transfer matrices along the linac.

        Returns
        -------
        is_3d : bool
            If the simulation is in 3D or not.
        n_points : int
            Number of mesh points along the linac.
        cumulated : np.ndarray
            Cumulated transfer matrices.

        """
        if cumulated is None:
            logging.error("You must provide at least one of the two "
                          "arrays: individual transfer matrices or "
                          "cumulated transfer matrices.")
            raise IOError("Wrong input")
        is_3d = self._determine_if_is_3d(cumulated)
        n_points = cumulated.shape[0]
        return is_3d, n_points, cumulated

    def _compute_cumulated(self,
                           is_3d: bool,
                           n_points: int) -> np.ndarray:
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

        .. todo::
            I think the 3D/1D handling may be smarter?

        """
        fill_value = np.NaN
        shape = (n_points, 2, 2)
        if is_3d:
            fill_value = 0.
            shape = (n_points, 6, 6)
        cumulated = np.full(shape, fill_value)

        cumulated[0] = self.individual[0]
        for i in range(1, n_points):
            cumulated[i] = self.individual[i] @ cumulated[i - 1]
        if is_3d:
            return cumulated

        cumulated_1d = cumulated
        cumulated = np.full((n_points, 6, 6), fill_value)
        cumulated[:, 4:, 4:] = cumulated_1d
        return cumulated

    def _determine_if_is_3d(self, array: np.ndarray) -> bool:
        shape = array.shape[1:]
        if shape == (2, 2):
            return False
        if shape == (6, 6):
            return True
        logging.error(f"The individual transfer matrices have shape {shape}, "
                      "while (2, 2) (1D) or (6, 6) (3D) are expected.")
        raise IOError("Wrong dimensions for given transfer matrices.")

    def _insert_eye_transfer_matrix_at_start(self, array: np.ndarray
                                             ) -> np.ndarray:
        """Insert eye matrix at first position of ``array``."""
        self.n_points += 1
        eye = np.eye(6)
        return np.vstack((eye[np.newaxis], array))

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
