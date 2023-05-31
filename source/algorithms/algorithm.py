#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:42:14 2023.

@author: placais

Template class for the optimisation algorithms.
"""
from typing import Callable, Any
from dataclasses import dataclass
import numpy as np

from optimisation.variables import VariablesAndConstraints


@dataclass
class OptimisationAlgorithm:
    """Holds the optimisation parameters, the methods to optimize."""

    variables_constraints: VariablesAndConstraints
    compute_residuals: Callable[[dict], np.ndarray]
    solution: dict = {}

    def optimise(self) -> tuple[bool, dict[str, list[float] | None]]:
        """
        Set up optimisation parameters and solve the problem.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        info : dict[str, list[float]]] | None
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.
        """
        success = False
        info = {'X': None, 'F': None, 'G': None}
        return success, info

    def _format_variables_and_constraints(self) -> Any:
        """
        Transform generic VariableAndConstraints.

        Output must be understandable by the optimisation algorithm that is
        used.
        """
        return self.variables_constraints

    def _wrapper(self):
        """Modify input variables and call the transfer matrices function."""
        pass
