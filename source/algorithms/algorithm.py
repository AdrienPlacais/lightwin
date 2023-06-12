#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:42:14 2023.

@author: placais

Template class for the optimisation algorithms.

Abstract methods are mandatory and a TypeError will be raised if you try to
create your own algorithm and omit them.
"""
from typing import Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from optimisation.variables import VariablesAndConstraints
from optimisation.set_of_cavity_settings import SetOfCavitySettings
from beam_calculation.output import SimulationOutput
from core.elements import FieldMap


@dataclass
class OptimisationAlgorithm(ABC):
    """Holds the optimisation parameters, the methods to optimize."""

    variables_constraints: VariablesAndConstraints
    compute_residuals: Callable[[SimulationOutput], np.ndarray]
    compute_beam_propagation: Callable[[SetOfCavitySettings, bool],
                                       SimulationOutput]
    compensating_cavities: list[FieldMap]
    variable_names: list[str]
    phi_s_fit: bool

    def __post_init__(self) -> None:
        """Set the output object."""
        self.solution: SetOfCavitySettings

    @abstractmethod
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

    @abstractmethod
    def _format_variables_and_constraints(self) -> Any:
        """
        Transform generic VariableAndConstraints.

        Output must be understandable by the optimisation algorithm that is
        used.
        """

    @abstractmethod
    def _wrapper_residuals(self, var: np.ndarray) -> np.ndarray:
        """
        Compute the residuals.

        In particular: allow the optimisation algorithm to communicate with the
        beam propagation function (compute_transfer_matrices), and convert the
        results of the beam propagation function to residuals.
        """

    @abstractmethod
    def _create_set_of_cavity_settings(self, var: np.ndarray
                                       ) -> SetOfCavitySettings:
        """
        Make generic the `var`, specific to each optimisation algorithm.

        Also very useful to avoid mixing up the norms and phases between the
        different cavities.
        """
