#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:01:48 2023.

@author: placais

This module holds :class:`LeastSquares`, a simple and fast optimisation method.

"""
from dataclasses import dataclass
import logging

from scipy.optimize import least_squares, Bounds
import numpy as np

from optimisation.algorithms.algorithm import OptimisationAlgorithm
from failures.set_of_cavity_settings import SetOfCavitySettings


@dataclass
class LeastSquares(OptimisationAlgorithm):
    """
    Plain least-squares method, efficient for small problems.

    It does not support constraints. One approach to put constraints on the
    synchronous phase is to set ``phi_s_fit`` to ``True``, making the
    synchronous phase a bounded variable.

    Notes
    -----
    Works very well with :class:`Envelope1D`, has issues converging with
    :class:`TraceWin`.

    All the attributes but ``solution`` are inherited from the Abstract Base
    Class :class:`OptimisationAlgorithm`.

    See also
    --------
    :class:`LeastSquaresPenalty`

    """

    def __post_init__(self) -> None:
        """Set additional information."""
        self.supports_constraints = False

    def optimise(self) -> tuple[bool,
                                SetOfCavitySettings,
                                dict[str, list[float]]]:
        """
        Set up the optimisation and solve the problem.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        optimized_cavity_settings : SetOfCavitySettings
            Best solution found by the optimization algorithm.
        info : dict[str, list[float]]] | None
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.

        """
        kwargs = self._algorithm_parameters()
        x_0, bounds = self._format_variables()

        solution = least_squares(fun=self._wrapper_residuals,
                                 x0=x_0,
                                 bounds=bounds,
                                 **kwargs)

        self.solution = solution
        optimized_cavity_settings = self._create_set_of_cavity_settings(
            solution.x)
        # TODO: output some info could be much more clear by using the __str__
        # methods of the various objects.

        self._output_some_info()

        success = self.solution.success
        info = {'X': self.solution.x.tolist(),
                'F': self.solution.fun.tolist(),
                }
        return success, optimized_cavity_settings, info

    def _algorithm_parameters(self) -> dict:
        """Create the ``kwargs`` for the optimisation."""
        kwargs = {'jac': '2-point',     # Default
                  # 'trf' not ideal as jac is not sparse. 'dogbox' may have
                  # difficulties with rank-defficient jacobian.
                  'method': 'dogbox',
                  'ftol': 1e-10,
                  'gtol': 1e-8,
                  'xtol': 1e-8,
                  # 'x_scale': 'jac',
                  # 'loss': 'arctan',
                  'diff_step': None, 'tr_solver': None, 'tr_options': {},
                  'jac_sparsity': None,
                  'verbose': 0,
                  }
        return kwargs

    def _format_variables(self) -> tuple[np.ndarray, Bounds]:
        """Convert the :class:`Variable`s to an array and :class:`Bounds`."""
        x_0 = np.array([var.x_0 for var in self.variables])
        _bounds = np.array([var.limits for var in self.variables])
        bounds = Bounds(_bounds[:, 0], _bounds[:, 1])
        return x_0, bounds

    def _output_some_info(self) -> None:
        """Show the most useful data from :func:`least_squares`."""
        sol = self.solution
        info_string = "Objective functions results:\n"
        for i, fun in enumerate(sol.fun):
            info_string += f"{i}: {' ':>35} | {fun}\n"
        logging.info(info_string)
        info_string = "least_squares algorithm output:"
        info_string += f"\nmessage: {sol.message}\n"
        info_string += f"nfev: {sol.nfev}\tnjev: {sol.njev}\n"
        info_string += f"optimality: {sol.optimality}\nstatus: {sol.status}\n"
        info_string += f"success: {sol.success}\nsolution: {sol.x}\n"
        logging.debug(info_string)
