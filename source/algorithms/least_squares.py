#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:01:48 2023.

@author: placais

Class to solve the problem by the least_squares method. Attributes are
inherited from the parent OptimisationAlgorithm class:
    variables_constraints: VariablesAndConstraints
    compute_residuals: Callable[[dict], np.ndarray]
"""
from dataclasses import dataclass
import logging

from scipy.optimize import least_squares, Bounds
import numpy as np

from algorithms.algorithm import OptimisationAlgorithm


@dataclass
class LeastSquares(OptimisationAlgorithm):
    """Plain least-squares method, efficient for small problems."""

    def optimise(self) -> tuple[bool, dict[str, list[float]]]:
        """
        Set up the optimisation and solve the problem.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        info : dict[str, list[float]]] | None
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.
        """
        kwargs = {'jac': '2-point',     # Default
                  # 'trf' not ideal as jac is not sparse. 'dogbox' may have
                  # difficulties with rank-defficient jacobian.
                  'method': 'dogbox',
                  'ftol': 1e-8, 'gtol': 1e-8,   # Default
                  # Solver is sometimes 'lazy' and ends with xtol
                  # termination condition, while settings are clearly not
                  #  optimized
                  'xtol': 1e-8,
                  # 'x_scale': 'jac',
                  # 'loss': 'arctan',
                  'diff_step': None, 'tr_solver': None, 'tr_options': {},
                  'jac_sparsity': None}
                  # 'verbose': debugs['verbose']}

        x_0, bounds = self._format_variables_and_constraints()

        self.solution = least_squares(
            fun=self._wrapper_residuals,
            x0=x_0, bounds=bounds,
            # args=None,
            **kwargs)
        self._output_some_info()

        success = self.solution.success
        info = {'X': self.solution.x.tolist(),
                'F': self.solution.fun.tolist(),
                }
        return success, info

    def _wrapper_residuals(self, var: np.ndarray):
        """Unpack arguments, compute residuals."""
        # Unpack arguments
        d_fits = {
            'l_phi': var[:var.size // 2].tolist(),
            'l_k_e': var[var.size // 2:].tolist(),
            'phi_s fit': self.phi_s_fit,
        }
        simulation_output = self.compute_beam_propagation(d_fits,
                                                          transfer_data=False)
        residuals = self.compute_residuals(simulation_output)
        return residuals

    def _format_variables_and_constraints(self
                                          ) -> tuple[np.ndarray, Bounds]:
        """Return design space as expected by scipy.least_squares."""
        x_0 = np.array([var.x_0
                        for var in self.variables_constraints.variables])
        _bounds = np.array([var.limits
                            for var in self.variables_constraints.variables])
        bounds = Bounds(_bounds[:, 0], _bounds[:, 1])
        return x_0, bounds

    def _output_some_info(self) -> None:
        """Show the most useful data from least_squares."""
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
