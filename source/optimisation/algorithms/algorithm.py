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

from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint
from failures.set_of_cavity_settings import SetOfCavitySettings

from beam_calculation.output import SimulationOutput

from core.list_of_elements import ListOfElements
from core.elements import FieldMap


# TODO check if it is necessary to pass out the whole elts to
# OptimisationAlgorithm?
@dataclass
class OptimisationAlgorithm(ABC):
    """
    Holds the optimisation parameters, the methods to optimize.

    Attributes
    ----------
    compensating_cavities : list[FieldMap]
        Cavity objects used to compensate for the faults.
    elts : ListOfElements
        Holds the whole compensation zone under study.
    solution : dict
        Holds information on the solution that was found.
    variables : list[Variable]
        Holds variables, their initial values, their limits.
    constraints : list[Constraint] | None, optional
        Holds constraints and their limits. The default is None.

    Methods
    -------
    compute_beam_propagation: Callable[SetOfCavitySettings, SimulationOutput]
        Method to compute propagation of the beam with the given cavity
        settings. Defined by a `BeamCalculator.run_with_this` method, the
        positional argument `elts` being set by a `functools.partial`.
    compute_residuals : Callable[SimulationOutput, Any]
        Method to compute residuals from a `SimulationOutput`.

    Abstract methods
    ----------------
    optimise : Callable[None, [bool,
                               SetOfCavitySettings,
                               dict[str, list[float] | None]
    _format_variables_and_constraints : Callable[None, Any]
    _create_set_of_cavity_settings : Callable[Any, SetOfCavitySettings]

    """

    compute_beam_propagation: Callable[SetOfCavitySettings, SimulationOutput]
    compute_residuals: Callable[SimulationOutput, np.ndarray]
    compensating_cavities: list[FieldMap]
    elts: ListOfElements
    variables: list[Variable]
    constraints: list[Constraint] | None = None
    compute_constraints: Callable[SimulationOutput, np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Set the output object."""
        self.solution: dict
        self.supports_constraints: bool

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
        residuals = self.compute_residuals(simulation_output=simulation_output)
        return residuals

    @abstractmethod
    def _create_set_of_cavity_settings(self, var: Any) -> SetOfCavitySettings:
        """
        Make generic the `var`, specific to each optimisation algorithm.

        Also very useful to avoid mixing up the norms and phases between the
        different cavities.

        """
