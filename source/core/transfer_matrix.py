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
                 first_cumulated_transfer_matrix: np.ndarray | None = None,
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
        first_cumulated_transfer_matrix : np.ndarray
            First transfer matrix. The default is None, in which case it will
            be converted to the eye matrix -- corresponds to the study of a
            full linac. It should be the cumulated transfer matrix of the
            previous portion of linac otherwise.

        """
        if isinstance(individual, list):
            individual = np.array(individual)

        self.individual: np.ndarray
        if individual is not None:
            self.individual = individual
            is_3d, n_points, cumulated = self._init_from_individual(
                individual,
                first_cumulated_transfer_matrix)

        else:
            is_3d, n_points, cumulated = self._init_from_cumulated(
                cumulated,
                first_cumulated_transfer_matrix)

        self.is_3d = is_3d
        self.n_points = n_points

        self.cumulated = cumulated

    def _init_from_individual(
            self,
            individual: np.ndarray,
            first_cumulated_transfer_matrix: np.ndarray | None,
            ) -> tuple[bool, int, np.ndarray]:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        individual : np.ndarray
            Individual transfer matrices along the linac.
        first_cumulated_transfer_matrix : np.ndarray | None
            First transfer matrix. It should be None if we study a linac
            from the start (``z_pos == 0.``), and should be the cumulated
            transfer matrix of the previous linac portion otherwise.

        Returns
        -------
        is_3d : bool
            If the simulation is in 3D or not.
        n_points : int
            Number of mesh points along the linac.
        cumulated : np.ndarray
            Cumulated transfer matrices.

        """
        n_points = individual.shape[0] + 1
        is_3d = self._determine_if_is_3d(individual)
        if is_3d:
            shape = (n_points, 6, 6)
        else:
            shape = (n_points, 2, 2)

        if first_cumulated_transfer_matrix is None:
            first_cumulated_transfer_matrix = np.eye(shape[1])

        cumulated = self._compute_cumulated(first_cumulated_transfer_matrix,
                                            shape,
                                            is_3d,
                                            n_points)
        return is_3d, n_points, cumulated

    def _init_from_cumulated(
            self,
            cumulated: np.ndarray | None,
            first_cumulated_transfer_matrix: np.ndarray | None,
            tol: float = 1e-8
            ) -> tuple[bool, int, np.ndarray]:
        """Check that the given cumulated matrix is valid.

        Parameters
        ----------
        cumulated : np.ndarray
            Cumulated transfer matrices along the linac.
        first_cumulated_transfer_matrix : np.ndarray | None
            The first of the cumulated transfer matrices. The default is None,
            in which case we insert an eye matrix at the first position.
            Otherwise, we insert the given matrix. This insertion is skipped if
            the matrix we try to insert is already the first transfer matrix.
        tol : float, optional
            The max allowed difference between ``cumulated`` and
            ``first_cumulated_transfer_matrix`` when determining if they are
            the same or not.

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

        if first_cumulated_transfer_matrix is None:
            first_cumulated_transfer_matrix = np.eye(6)

        if (np.abs(cumulated[0]
                   - first_cumulated_transfer_matrix)).any() > tol:
            n_points += 1
            cumulated = np.vstack((first_cumulated_transfer_matrix[np.newaxis],
                                   cumulated))

        return is_3d, n_points, cumulated

    def _compute_cumulated(self,
                           first_cumulated_transfer_matrix: np.ndarray,
                           shape: tuple[int, int, int],
                           is_3d: bool,
                           n_points: int) -> np.ndarray:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        first_cumulated_transfer_matrix : np.ndarray
            First transfer matrix. It should be eye matrix if we study a linac
            from the start (``z_pos == 0.``), and should be the cumulated
            transfer matrix of the previous linac portion otherwise.
        shape : tuple[int, int, int]
            Shape of the output ``cumulated`` array.
        is_3d : bool
            If the simulation is in 3D or not.
        n_points : int
            Number of mesh points along the linac.

        Returns
        -------
        cumulated : np.ndarray
            Cumulated transfer matrix.

        .. todo::
            I think the 3D/1D handling may be smarter?

        """
        cumulated = np.full(shape, np.NaN)
        cumulated[0] = first_cumulated_transfer_matrix

        for i in range(n_points - 1):
            cumulated[i + 1] = self.individual[i] @ cumulated[i]

        if is_3d:
            return cumulated

        cumulated_1d = cumulated
        cumulated = np.full((n_points, 6, 6), np.NaN)
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
