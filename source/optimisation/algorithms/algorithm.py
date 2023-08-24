#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:42:14 2023.

@author: placais

Template class for the optimisation algorithms.

Abstract methods are mandatory and a ``TypeError`` will be raised if you try to
create your own algorithm and omit them.

.. todo::
    Check if it is necessary to pass out the whole ``elts`` to
    ``OptimisationAlgorithm``?

"""
import logging
from typing import Callable, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np

from optimisation.parameters.objective import Objective
from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint
from failures.set_of_cavity_settings import (SetOfCavitySettings,
                                             SingleCavitySettings)

from beam_calculation.output import SimulationOutput

from core.list_of_elements import ListOfElements
from core.elements.elements.field_map import FieldMap


@dataclass
class OptimisationAlgorithm(ABC):
    """
    Holds the optimisation parameters, the methods to optimize.

    Parameters
    ----------
    compensating_cavities : list[FieldMap]
        Cavity objects used to compensate for the faults.
    elts : ListOfElements
        Holds the whole compensation zone under study.
    objectives : list[Objective]
        Holds objectives, initial values, bounds.
    variables : list[Variable]
        Holds variables, their initial values, their limits.
    constraints : list[Constraint] | None, optional
        Holds constraints and their limits. The default is None.
    solution : dict
        Holds information on the solution that was found.
    supports_constraints : bool
        If the method handles constraints or not.
    compute_beam_propagation: Callable[[SetOfCavitySettings], SimulationOutput]
        Method to compute propagation of the beam with the given cavity
        settings. Defined by a :func:`BeamCalculator.run_with_this` method, the
        positional argument ``elts`` being set by a :func:`functools.partial`.
    compute_residuals : Callable[[SimulationOutput], Any]
        Method to compute residuals from a :class:`SimulationOutput`.
    compute_constraints : Callable[[SimulationOutput], np.ndarray] | None, optional
        Method to compute constraint violation. The default is None.
    """

    compensating_cavities: list[FieldMap]
    elts: ListOfElements

    objectives: list[Objective]
    variables: list[Variable]
    compute_beam_propagation: Callable[[SetOfCavitySettings], SimulationOutput]
    compute_residuals: Callable[[SimulationOutput], np.ndarray]

    constraints: list[Constraint] | None = None
    compute_constraints: Callable[[SimulationOutput], np.ndarray] | None = None

    def __post_init__(self) -> None:
        """Set the output object."""
        self.solution: dict
        self.supports_constraints: bool

    @property
    def variable_names(self) -> list[str]:
        """Give name of all variables."""
        return [variable.name for variable in self.variables]

    @property
    def n_var(self) -> int:
        """Give number of variables."""
        return len(self.variables)

    @property
    def n_obj(self) -> int:
        """Give number of objectives."""
        return len(self.objectives)

    @property
    def n_constr(self) -> int:
        """Return number of (inequality) constraints."""
        return sum([constraint.n_constraints
                    for constraint in self.constraints])

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

    def _format_variables(self) -> Any:
        """Transform all :class:`Variable`s for this optimisation algorithm.

        """

    def _format_objectives(self) -> Any:
        """Transform all :class:`Objective`s for this optimisation algorithm.

        """

    def _format_constraints(self) -> Any:
        """Transform all :class:`Constraint`s for this optimisation algorithm.

        """

    def _wrapper_residuals(self, var: np.ndarray) -> np.ndarray:
        """Compute residuals from an array of variable values."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output=simulation_output)
        return residuals

    def _norm_wrapper_residuals(self, var: np.ndarray) -> float:
        """Compute norm of residues vector from an array of variable values."""
        return np.linalg.norm(self._wrapper_residuals(var))

    def _create_set_of_cavity_settings(self, var: Any) -> SetOfCavitySettings:
        """
        Make generic the ``var``, specific to each optimisation algorithm.

        Also very useful to avoid mixing up the norms and phases between the
        different cavities.

        """
        my_phi = list(var[:var.shape[0] // 2])
        my_ke = list(var[var.shape[0] // 2:])
        my_vars = zip(self.compensating_cavities, my_ke, my_phi)

        if 'phi_s' in self.variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_s=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in my_vars]
            return SetOfCavitySettings(my_set)

        if 'phi_0_abs' in self.variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_0_abs=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in my_vars]
            return SetOfCavitySettings(my_set)

        if 'phi_0_rel' in self.variable_names:
            my_set = [SingleCavitySettings(cavity=cavity,
                                           k_e=k_e,
                                           phi_0_rel=phi,
                                           index=self.elts.index(cavity))
                      for cavity, k_e, phi in my_vars]
            return SetOfCavitySettings(my_set)

        logging.critical("Error in the _create_set_of_cavity_settings")
        return None
