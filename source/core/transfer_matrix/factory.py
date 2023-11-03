#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provide an easy way to generate :class:`.TransferMatrix`."""


from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from core.transfer_matrix.transfer_matrix import TransferMatrix


class TransferMatrixFactory(ABC):
    """Provide a method for easy creation of :class:`.TransferMatrix`.

    This class should be subclassed by every :class:`.BeamCalculator`.

    """

    def __init__(self,
                 is_3d: bool,
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

    def _preprocess(self, *args, **kwargs) -> None:
        """Preprocess the data given by the :class:`.BeamCalculator`."""
        return

    @abstractmethod
    def run(self, *args, **kwargs) -> TransferMatrix:
        """Create the transfer matrix from a simulation.

        Returns
        -------
        TransferMatrix
            Holds all cumulated transfer matrices in all the planes.

        """
        self._preprocess(*args, **kwargs)
        transfer_matrix = TransferMatrix(*args, **kwargs)
        return transfer_matrix
