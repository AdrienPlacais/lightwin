#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds a variation of :class:`LeastSquares`.

It is not intended to be used with ``phi_s fit``. Approach is here to make the
residues grow when the constraints are not respected.

"""
from dataclasses import dataclass
import logging

import numpy as np

from optimisation.algorithms.least_squares import LeastSquares


@dataclass
class LeastSquaresPenalty(LeastSquares):
    """
    A least-squares method, with a penalty function to consider constraints.

    Everything is inherited from :class:`LeastSquares`.

    """

    def __post_init__(self) -> None:
        """Set additional information."""
        self.supports_constraints = True

        if 'phi_s' in self.variable_names:
            logging.error("This algorithm is not intended to work with synch "
                          "phase as variables, but rather as constraint.")

    def _wrapper_residuals(self, var: np.ndarray) -> np.array:
        """Give residuals with a penalty."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output=simulation_output)
        constraints_evaluations = self.compute_constraints(simulation_output)
        penalty = self._penalty(constraints_evaluations)

        return residuals * penalty

    def _penalty(self, constraints_evaluations: np.ndarray) -> float:
        """Compute appropriate penalty."""
        violated_constraints = constraints_evaluations[
            np.where(constraints_evaluations > 0.)]
        n_violated = violated_constraints.shape[0]
        if n_violated == 0:
            return 1.
        return 1. + np.sum(n_violated) * 10.
