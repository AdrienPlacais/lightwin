#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:01:48 2023.

@author: placais

This module holds :class:`Explorator`, a module to explorate the whole design
space. In order to be consistent with the ABC :class:`OptimisationAlgorithm`,
it also returns the solution with the lowest residue value -- hence it is also
a "brute-force" optimisation algorith.

"""
import logging
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

from optimisation.algorithms.algorithm import OptimisationAlgorithm
from failures.set_of_cavity_settings import SetOfCavitySettings
from util.dicts_output import markdown


@dataclass
class Explorator(OptimisationAlgorithm):
    """
    Method that tries all the possible solutions.

    Notes
    -----
    Very inefficient for optimisation. It is however useful to study a specific
    case.

    All the attributes but ``solution`` are inherited from the Abstract Base
    Class :class:`OptimisationAlgorithm`.

    """

    def __post_init__(self) -> None:
        """Set additional information."""
        self.supports_constraints = True

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
        self._check_dimensions()
        kwargs = self._algorithm_parameters()
        variable_mesh, variables_comb = self._generate_combinations(**kwargs)

        results = [self._wrapper_residuals(var) for var in variables_comb]
        objective_values = np.array([res[0] for res in results])
        are_constraints_ok = np.array([res[1] for res in results])

        objective_mesh = self._results_as_mesh(objective_values, **kwargs)
        constraint_mesh = self._results_as_mesh(are_constraints_ok, **kwargs)

        axes = self._output_some_info(variable_mesh, objective_mesh,
                                      constraint_mesh)

        best_idx = np.nanargmin(objective_values)
        info = {'X': variables_comb[best_idx],
                'F': objective_values[best_idx]
                }

        self._add_the_best_point(axes, info['X'], info['F'])

        optimized_cavity_settings = self._create_set_of_cavity_settings(
            info['X'])
        return True, optimized_cavity_settings, info

    def _check_dimensions(self) -> None:
        """Check that we have proper number of vars and objectives."""
        if self.n_obj != 1:
            logging.warning("The number of objectives is different from 1. "
                            "Hence I will simply plot the norm of objectives.")
        assert self.n_var == 2, "Wrong number of variables, cannot represent."

    def _algorithm_parameters(self) -> dict:
        """Create the ``kwargs`` for the optimisation."""
        kwargs = {'n_points': 10}
        return kwargs

    def _generate_combinations(self, n_points: int = 10, **kwargs
                               ) -> tuple[np.ndarray, np.ndarray]:
        """Generate all the possible combinations of the variables."""
        limits = []
        for var in self.variables:
            lim = (var.limits[0], var.limits[1])

            if 'phi' in var.name and lim[1] - lim[0] >= 2. * np.pi:
                lim = (0., 2. * np.pi)
            limits.append(lim)

        variables_values = [np.linspace(lim[0], lim[1], n_points)
                            for lim in limits]
        variables_mesh = np.array(np.meshgrid(*variables_values,
                                              indexing='ij'))
        variables_combinations = np.concatenate(variables_mesh.T)
        return variables_mesh, variables_combinations

    def _wrapper_residuals(self, var: np.ndarray
                           ) -> tuple[float, bool]:
        """Give norm of residual and if phi_s constraint is respected."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output=simulation_output)
        constraints_evaluations = self.compute_constraints(simulation_output)
        is_ok = constraints_evaluations[0] < 0. \
            and constraints_evaluations[1] < 0.
        return np.linalg.norm(residuals), is_ok

    def _results_as_mesh(self, objective_values: np.ndarray,
                         n_points: int = 10, **kwargs) -> np.ndarray:
        """Reformat the results for plotting purposes."""
        return objective_values.reshape((n_points, n_points)).T

    def _output_some_info(self,
                          variable_mesh: np.ndarray,
                          objective_mesh: np.ndarray,
                          constraint_mesh: np.ndarray) -> mplot3d.Axes3D:
        """Plot the design space."""
        fig = plt.figure(30)
        axes = fig.add_subplot(projection='3d')
        axes.set_xlabel(markdown[self.variable_names[0]].replace('deg', 'rad'))
        axes.set_ylabel(markdown[self.variable_names[1]].replace('deg', 'rad'))
        axes.set_zlabel('Objective')

        colors = cm.Set1(constraint_mesh)
        rcount, ccount, _ = colors.shape
        axes.plot_wireframe(variable_mesh[0],
                            variable_mesh[1],
                            objective_mesh,
                            rcount=rcount,
                            ccount=ccount,
                            )
        for i in range(len(variable_mesh[0])):
            axes.scatter(variable_mesh[0][i],
                         variable_mesh[1][i],
                         objective_mesh[i],
                         s=50,
                         c=colors[i])
        plt.show()
        return axes

    def _add_the_best_point(self, axes: mplot3d.Axes3D, var: np.ndarray,
                            obj: np.ndarray) -> None:
        """Add the best solution to the plot."""
        axes.scatter3D(var[0], var[1], obj, s=100, c='r')
