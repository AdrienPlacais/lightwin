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

from failures.variables import VariablesAndConstraints
from failures.set_of_cavity_settings import SetOfCavitySettings

from beam_calculation.output import SimulationOutput

from core.list_of_elements import ListOfElements
from core.elements import FieldMap


# TODO check if variable_names could be in variables_constraints?
# TODO check if it is necessary to pass out the whole elts to
# OptimisationAlgorithm?
@dataclass
class OptimisationAlgorithm(ABC):
    """
    Holds the optimisation parameters, the methods to optimize.

    Attributes
    ----------
    variables_constraints : VariablesAndConstraints
        Holds the initial value and bounds of the variables, as well as the
        bounds for the constraints.
    compensating_cavities : list[FieldMap]
        Cavity objects used to compensate for the faults.
    variable_names : list[str]
        Name of the variables.
    elts : ListOfElements
        Holds the whole compensation zone under study.
    solution : dict
        Holds information on the solution that was found.

    Methods
    -------
    compute_beam_propagation: Callable[SetOfCavitySettings, SimulationOutput]
        Method to compute propagation of the beam with the given cavity
        settings. Defined by a `BeamCalculator.run_with_this` method, the
        positional argument `elts` being set by a `functools.partial`.
    compute_residuals : Callable[SimulationOutput, np.ndarray]
        Method to compute residuals from a `SimulationOutput`.

    Abstract methods
    ----------------
    optimise : Callable[None, [bool,
                               SetOfCavitySettings,
                               dict[str, list[float] | None]
    _format_variables_and_constraints : Callable[None, Any]
    _create_set_of_cavity_settings : Callable[np.ndarray, SetOfCavitySettings]

    """

    variables_constraints: VariablesAndConstraints
    compute_beam_propagation: Callable[SetOfCavitySettings, SimulationOutput]
    compute_residuals: Callable[SimulationOutput, np.ndarray]
    compensating_cavities: list[FieldMap]
    variable_names: list[str]
    elts: ListOfElements

    def __post_init__(self) -> None:
        """Set the output object."""
        self.solution: dict

    @abstractmethod
    def optimise(self) -> tuple[bool,
                                SetOfCavitySettings,
                                dict[str, list[float]] | None]:
        """
        Set up optimisation parameters and solve the problem.

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

    @abstractmethod
    def _format_variables_and_constraints(self) -> Any:
        """
        Transform generic VariableAndConstraints.

        Output must be understandable by the optimisation algorithm that is
        used.

        """

    def _wrapper_residuals(self, var: np.ndarray) -> np.ndarray:
        """Compute residuals from an array of variable values."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output)
        return residuals

    @abstractmethod
    def _create_set_of_cavity_settings(self, var: np.ndarray
                                       ) -> SetOfCavitySettings:
        """
        Make generic the `var`, specific to each optimisation algorithm.

        Also very useful to avoid mixing up the norms and phases between the
        different cavities.

        """
