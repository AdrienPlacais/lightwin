#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 16:30:10 2023.

@author: placais

This modules holds :class:`DownhillSimplex`, also called Nelder-Mead algorithm.

"""
from dataclasses import dataclass
import logging

from scipy.optimize import minimize, Bounds
import numpy as np

from optimisation.algorithms.algorithm import OptimisationAlgorithm
from failures.set_of_cavity_settings import (SetOfCavitySettings,
                                             SingleCavitySettings)


@dataclass
class DownhillSimplex(OptimisationAlgorithm):
    """
    Downhill simplex method, which does not use derivatives.

    All the attributes but ``solution`` are inherited from the Abstract Base
    Class :class:`OptimisationAlgorithm`.

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
        kwargs = {'method': 'Nelder-Mead',
                  'options': {
                      'adaptive': True,
                      'disp': True,
                  },
                  }

        x_0, bounds = self._format_variables_and_constraints()

        solution = minimize(fun=self._norm_wrapper_residuals,
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

    def _format_variables_and_constraints(self
                                          ) -> tuple[np.ndarray, Bounds]:
        """Return design space as expected by :func:`scipy.minimize`."""
        x_0 = np.array([var.x_0
                        for var in self.variables])
        _bounds = np.array([var.limits
                            for var in self.variables])
        bounds = Bounds(_bounds[:, 0], _bounds[:, 1])
        return x_0, bounds

    def _create_set_of_cavity_settings(self, var: np.ndarray
                                       ) -> SetOfCavitySettings:
        """Transform the array given by algorithm to a generic object."""
        # FIXME
        my_phi = list(var[:var.shape[0] // 2])
        my_ke = list(var[var.shape[0] // 2:])

        variable_names = [variable.name for variable in self.variables]

        if 'phi_s' in variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_s=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in zip(self.compensating_cavities,
                                                  my_ke,
                                                  my_phi)]
        elif 'phi_0_abs' in variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_0_abs=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in zip(self.compensating_cavities,
                                                  my_ke,
                                                  my_phi)]
        elif 'phi_0_rel' in variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_0_rel=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in zip(self.compensating_cavities,
                                                  my_ke,
                                                  my_phi)]
        else:
            logging.critical("Error in the _create_set_of_cavity_settings")
            return None

        my_set = SetOfCavitySettings(my_set)
        return my_set

    def _output_some_info(self) -> None:
        """Show the most useful data from least_squares."""
        sol = self.solution
        info_string = "Objective functions results:\n"
        objectives = self._wrapper_residuals(sol.x)
        for i, fun in enumerate(objectives):
            info_string += f"{i}: {' ':>35} | {fun}\n"
        info_string += f"Norm: {sol.fun}"
        logging.info(info_string)
        info_string = "Nelder-Mead algorithm output:"
        info_string += f"\nmessage: {sol.message}\n"
        # info_string += f"nfev: {sol.nfev}\tnjev: {sol.njev}\n"
        # info_string += f"optimality: {sol.optimality}\nstatus: {sol.status}\n"
        info_string += f"success: {sol.success}\nsolution: {sol.x}\n"
        logging.debug(info_string)
