#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide an easy way to generate :class:`.TransferMatrix`."""


from abc import ABC
from typing import Callable

import numpy as np

from core.transfer_matrix.transfer_matrix import TransferMatrix


class TransferMatrixFactory(ABC):
    """Provide a method for easy creation of :class:`.TransferMatrix`.

    This class should be an attribute of every :class:`.ListOfElements`.

    """

    def __init__(self,
                 is_3d: bool,
                 first_cumulated_transfer_matrix: np.ndarray | None = None,
                 ) -> None:
        """Store the first cumulated transfer matrix.

        This quantity is a constant of the :class:`.ListOfElements`.

        first_cumulated_transfer_matrix : np.ndarray
            First transfer matrix. The default is None, in which case it will
            be converted to the eye matrix -- corresponds to the study of a
            full linac. It should be the cumulated transfer matrix of the
            previous portion of linac otherwise.

        """
        self.is_3d = is_3d
        if first_cumulated_transfer_matrix is None:
            first_cumulated_transfer_matrix = \
                self._eye_matrix_with_proper_shape()
        self.first_cumulated_transfer_matrix = first_cumulated_transfer_matrix

    def _eye_matrix_with_proper_shape(self) -> np.ndarray:
        """Give initial transfer matrix with good shape.

        Returns
        -------
        np.ndarray
            Eye matrix. Shape is (6, 6) if the simulation is in 3D, or (2, 2)
            otherwise.

        """
        if self.is_3d:
            return np.eye((6, 6))
        return np.eye((2, 2))

    def run(self,
            individual: np.ndarray | list[np.ndarray] | None = None,
            cumulated: np.ndarray | None = None,
            element_to_index: Callable | None = None
            ) -> TransferMatrix:
        """Create the transfer matrix from a simulation.

        Parameters
        ----------
        individual : np.ndarray | list[np.ndarray] | None
            Individual transfer matrices. The default is None, in which case
            the ``cumulated`` transfer matrix must be provided directly.
        cumulated : np.ndarray | None
            Cumulated transfer matrices. The default is None, in which case the
            ``individual`` transfer matrices must be given.
        element_to_index : Callable | None
            element_to_index

        Returns
        -------
        TransferMatrix
            Holds all cumulated transfer matrices in all the planes.

        .. todo::
            Simpler TransferMatrix object, more data transformation here?

        """
        transfer_matrix = TransferMatrix(
            individual=individual,
            cumulated=cumulated,
            first_cumulated_transfer_matrix=self.first_cumulated_transfer_matrix,
            element_to_index=element_to_index,
            is_3d=self.is_3d)
        return transfer_matrix
