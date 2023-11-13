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
        """Store if simulation is in 3D or not.

        Parameters
        ----------
        is_3d : bool
            If the simulation is 3D or not.

        """
        self.is_3d = is_3d

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
