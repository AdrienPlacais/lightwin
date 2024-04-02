#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Template class for the optimisation algorithms.

Abstract methods are mandatory and a ``TypeError`` will be raised if you try to
create your own algorithm and omit them.

When you add you own optimisation algorithm, do not forget to add it to the
list of implemented algorithms in the :mod:`config.optimisation.algorithm`.

.. todo::
    Check if it is necessary to pass out the whole ``elts`` to
    ``OptimisationAlgorithm``?

.. todo::
    Methods and flags to keep the optimisation history or not, and also to save
    it or not. See :class:`Explorator`.

.. todo::
    Better handling of the attribute ``folder``. In particular, a correct value
    should be set at the ``OptimisationAlgorithm`` instanciation.

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.elements.element import Element
from core.elements.field_maps.cavity_settings import CavitySettings
from core.elements.field_maps.cavity_settings_factory import ICavitySettingsFactory
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.variable import Variable
from optimisation.objective.objective import Objective

@dataclass
class OptimisationAlgorithm(ABC):
    """
    Holds the optimisation parameters, the methods to optimize.

    Parameters
    ----------
    compensating_elements : list[Element]
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
        Method to compute propagation of the beam with the given settings.
        Defined by a :func:`BeamCalculator.run_with_this` method, the
        positional argument ``elts`` being set by a :func:`functools.partial`.
    compute_residuals : Callable[[SimulationOutput], Any]
        Method to compute residuals from a :class:`SimulationOutput`.
    compute_constraints : Callable[[SimulationOutput], np.ndarray] | None, \
optional
        Method to compute constraint violation. The default is None.
    folder : str | None, optional
        Where history, phase space and other optimisation information will be
        saved if necessary. The default is None.
    cavity_settings_factory : ICavitySettingsFactory
        A factory to easily create the cavity settings to try at each iteration
        of the optimisation algorithm.

    """

    compensating_elements: list[Element]
    elts: ListOfElements

    objectives: list[Objective]
    variables: list[Variable]
    compute_beam_propagation: Callable[[SetOfCavitySettings], SimulationOutput]
    compute_residuals: Callable[[SimulationOutput], np.ndarray]

    constraints: list[Constraint] | None = None
    compute_constraints: Callable[[SimulationOutput], np.ndarray] | None = None

    folder: str | None = None

    cavity_settings_factory: ICavitySettingsFactory | None = None

    def __post_init__(self) -> None:
        """Set the output object."""
        assert all([elt.can_be_retuned for elt in self.compensating_elements])
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

    def _create_set_of_cavity_settings(self,
                                       var: np.ndarray,
                                       status='compensate (in progress)',
                                       ) -> SetOfCavitySettings | None:
        """Transform ``var`` into generic :class:`SetOfCavitySettings`."""
        reference = [x for x in self.variable_names if 'phi' in x][0]
        original_settings = [cavity.cavity_settings
                             for cavity in self.compensating_elements]
        freq_cavities_mhz = [settings.freq_cavity_mhz
                             for settings in original_settings]
        transf_mat_func_wrappers = [settings.transf_mat_func_wrappers
                                    for settings in original_settings]

        assert self.cavity_settings_factory is not None
        several_cavity_settings = self.cavity_settings_factory.\
            from_optimisation_algorithm(
                var,
                reference,
                freq_cavities_mhz,
                status=status,
                transf_mat_func_wrappers=transf_mat_func_wrappers,
            )
        return SetOfCavitySettings.for_new(several_cavity_settings,
                                           self.compensating_elements)

    def _get_objective_values(self) -> dict[str, float]:
        """Save the full array of objective values."""
        sol = self.solution
        objectives_values = self._wrapper_residuals(sol.x)
        objectives_values = {
            objective.name: objective_value
            for objective, objective_value
            in zip(self.objectives, objectives_values)
        }
        return objectives_values
