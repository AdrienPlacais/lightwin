#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 11:57:32 2023.

@author: placais

This module holds `NSGA`, a genetic algorithm for optimisation.

"""
from dataclasses import dataclass
from typing import Callable
import logging

import numpy as np

from pymoo.core.problem import ElementwiseProblem
from pymoo.core.algorithm import Algorithm
from pymoo.core.result import Result
from pymoo.optimize import minimize

from optimisation.algorithms.algorithm import OptimisationAlgorithm
from failures.set_of_cavity_settings import (SetOfCavitySettings,
                                             SingleCavitySettings)
from failures.variables import VariablesAndConstraints


@dataclass
class NSGA(OptimisationAlgorithm):
    """
    Non-dominated Sorted Genetic Algorithm, an algorithm handling constraints.

    All the attributes but `solution` are inherited from the Abstract Base
    Class `OptimisationAlgorithm`.

    """

    compute_constraints: Callable | None = None

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
        problem = MyElementwiseProblem(
            _wrapper_residuals=self._wrapper_residuals,
            **self._problem_arguments)

        algorithm = self._set_algorithm()

        result: Result = minimize(problem=problem,
                                  algorithm=algorithm,
                                  termination=None,
                                  seed=None,
                                  verbose=False,
                                  display=None,
                                  callback=None,
                                  return_least_infeasible=False,
                                  save_history=False)

        success = True
        set_of_cavity_settings = self._create_set_of_cavity_settings(result)
        info = {}
        return success, set_of_cavity_settings, info

    @property
    def _problem_arguments(self) -> dict[str, int | np.ndarray]:
        """Gather arguments required for `ElementwiseProblem`."""
        kwargs = {'n_var': self._n_var,
                  'n_obj': self._n_obj,
                  'n_ieq_constraints': self._n_ieq_constraints,
                  'xl': self._xl,
                  'xu': self._xu}
        return kwargs

    @property
    def _n_var(self) -> int:
        """Number of variables."""
        return len(self.variables_constraints.variables)

    @property
    def _n_obj(self) -> int:
        """Number of objectives."""
        logging.warning("Number of objectives manually set.")
        return 3

    @property
    def _n_ieq_constraints(self) -> int:
        """Number of inequality constraints."""
        logging.warning("Number of constraints manually set.")
        return len(self.variables_constraints.constraints) * 2

    @property
    def _xl(self) -> np.ndarray:
        """Return variables lower limits."""
        lower = [var.limits[0] for var in self.variables_constraints.variables]
        return np.array(lower)

    @property
    def _xu(self) -> np.ndarray:
        """Return variables upper limits."""
        upper = [var.limits[1] for var in self.variables_constraints.variables]
        return np.array(upper)

    def _wrapper_residuals(self, var: np.ndarray) -> tuple[np.ndarray,
                                                           np.ndarray]:
        """Compute residuals from an array of variable values."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)

        # residuals = self.compute_residuals(simulation_output)

        objective = self.compute_residuals(simulation_output)
        constraints = self.compute_constraints(simulation_output)
        return objective, constraints


    def _set_algorithm(self) -> Algorithm:
        """Set `pymoo`s `Algorithm` object."""
        algorithm = Algorithm()
        return algorithm

    def _create_set_of_cavity_settings(self, result: Result
                                       ) -> SetOfCavitySettings:
        """Transform the object given by NSGA to a generic object."""
        set_of_cavity_settings = result.f
        return set_of_cavity_settings

    def _format_variables_and_constraints(self) -> None:
        """Legacy?"""
        pass


class MyElementwiseProblem(ElementwiseProblem):
    """A first test implementation, eval single solution at a time."""

    def __init__(self,
                 _wrapper_residuals: Callable[np.ndarray, np.ndarray],
                 **kwargs: int | np.ndarray) -> None:
        """Create object."""
        self._wrapper_residuals = _wrapper_residuals
        super().__init__(**kwargs)

    def _evaluate(self, x: np.ndarray, out: dict[str, np.ndarray],
                  *args, **kwargs) -> dict[str, np.ndarray]:
        """Calculate and return the objectives."""
        out['F'], out['G'] = self._wrapper_residuals(x)
        return out
