"""Define the Abstract Base Class of optimisation algorithms.

Abstract methods are mandatory and a ``TypeError`` will be raised if you try to
create your own algorithm and omit them.

When you add you own optimisation algorithm, do not forget to add it to the
list of implemented algorithms in the :mod:`config.optimisation.algorithm`.

.. todo::
    Check if it is necessary to pass out the whole ``elts`` to
    ``OptimisationAlgorithm``?

.. todo::
    Methods and flags to keep the optimisation history or not, and also to save
    it or not. See :class:`.Explorator`.

.. todo::
    Better handling of the attribute ``folder``. In particular, a correct value
    should be set at the ``OptimisationAlgorithm`` instanciation.

"""

from abc import ABC, abstractmethod
from collections.abc import Collection
from pathlib import Path
from typing import Any, Callable, TypedDict

import numpy as np

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from core.elements.field_maps.cavity_settings_factory import (
    CavitySettingsFactory,
)
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.variable import Variable
from optimisation.objective.objective import Objective


class OptiInfo(TypedDict):
    """Hold information on how optimisation went."""

    hist_X: np.ndarray
    hist_F: np.ndarray
    hist_G: np.ndarray


class OptiSol(TypedDict):
    """Hold information on the solution."""

    x: np.ndarray


ComputeBeamPropagationT = Callable[[SetOfCavitySettings], SimulationOutput]
ComputeResidualsT = Callable[[SimulationOutput], Any]
ComputeConstraintsT = Callable[[SimulationOutput], np.ndarray]


class OptimisationAlgorithm(ABC):
    """Holds the optimisation parameters, the methods to optimize.

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
    compute_beam_propagation: ComputeBeamPropagationT
        Method to compute propagation of the beam with the given settings.
        Defined by a :func:`BeamCalculator.run_with_this` method, the
        positional argument ``elts`` being set by a :func:`functools.partial`.
    compute_residuals : ComputeResidualsT
        Method to compute residuals from a :class:`SimulationOutput`.
    compute_constraints : ComputeConstraintsT | None, \
    optional
        Method to compute constraint violation. The default is None.
    folder : str | None, optional
        Where history, phase space and other optimisation information will be
        saved if necessary. The default is None.
    cavity_settings_factory : ICavitySettingsFactory
        A factory to easily create the cavity settings to try at each iteration
        of the optimisation algorithm.

    """

    supports_constraints: bool

    def __init__(
        self,
        compensating_elements: Collection[Element],
        elts: ListOfElements,
        objectives: Collection[Objective],
        variables: Collection[Variable],
        compute_beam_propagation: ComputeBeamPropagationT,
        compute_residuals: ComputeResidualsT,
        cavity_settings_factory: CavitySettingsFactory,
        constraints: Collection[Constraint] | None = None,
        compute_constraints: ComputeConstraintsT | None = None,
        folder: Path | None = None,
        optimisation_algorithm_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Instantiate the object."""
        assert all([elt.can_be_retuned for elt in compensating_elements])
        self.compensating_elements = compensating_elements
        self.elts = elts

        self.objectives = objectives
        self.variables = variables
        self.compute_beam_propagation = compute_beam_propagation
        self.compute_residuals = compute_residuals
        self.constraints = constraints

        if self.supports_constraints:
            assert compute_constraints is not None
        self.compute_constraints = compute_constraints
        self.folder = folder
        self.cavity_settings_factory = cavity_settings_factory

        self.solution: OptiSol
        self.supports_constraints: bool

        if optimisation_algorithm_kwargs is None:
            optimisation_algorithm_kwargs = {}
        self.optimisation_algorithm_kwargs = (
            self._default_kwargs | optimisation_algorithm_kwargs
        )

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
        if self.constraints is None:
            return 0
        return sum(
            [constraint.n_constraints for constraint in self.constraints]
        )

    @property
    def _default_kwargs(self) -> dict[str, Any]:
        """Give the default optimisation algorithm kwargs."""
        return {}

    @abstractmethod
    def optimise(
        self,
        keep_history: bool = False,
        save_history: bool = False,
    ) -> tuple[bool, SetOfCavitySettings | None, OptiInfo]:
        """Set up optimisation parameters and solve the problem.

        Parameters
        ----------
        keep_history : bool, optional
            To keep all the variables that were tried as well as the associated
            objective and constraint violation values.
        save_history : bool, optional
            To save the history.

        Returns
        -------
        success : bool
            Tells if the optimisation algorithm managed to converge.
        optimized_cavity_settings : SetOfCavitySettings
            Best solution found by the optimization algorithm. None if no
            satisfactory solution was found.
        info : OptiInfo
            Gives list of solutions, corresponding objective, convergence
            violation if applicable, etc.

        """

    def _format_variables(self) -> Any:
        """Transform all :class:`Variable`s for this optimisation algorithm."""

    def _format_objectives(self) -> Any:
        """Transform all :class:`Objective`s for this optimisation algorithm."""

    def _format_constraints(self) -> Any:
        """Transform all :class:`Constraint`s for this optimisation algorithm."""

    def _wrapper_residuals(self, var: np.ndarray) -> np.ndarray:
        """Compute residuals from an array of variable values."""
        cav_settings = self._create_set_of_cavity_settings(var)
        simulation_output = self.compute_beam_propagation(cav_settings)
        residuals = self.compute_residuals(simulation_output)
        return residuals

    def _norm_wrapper_residuals(self, var: np.ndarray) -> float:
        """Compute norm of residues vector from an array of variable values."""
        return float(np.linalg.norm(self._wrapper_residuals(var)))

    def _create_set_of_cavity_settings(
        self,
        var: np.ndarray,
        status: str = "compensate (in progress)",
    ) -> SetOfCavitySettings:
        """Transform ``var`` into generic :class:`.SetOfCavitySettings`.

        Parameters
        ----------
        var
            An array holding the variables to try.
        status : str, optional
            mmmh

        Returns
        -------
        SetOfCavitySettings
            Object holding the settings of all the cavities.

        """
        reference = [x for x in self.variable_names if "phi" in x][0]
        original_settings = [
            cavity.cavity_settings for cavity in self.compensating_elements
        ]

        several_cavity_settings = (
            self.cavity_settings_factory.from_optimisation_algorithm(
                base_settings=original_settings,
                var=var,
                reference=reference,
                status=status,
            )
        )
        return SetOfCavitySettings.from_cavity_settings(
            several_cavity_settings, self.compensating_elements
        )

    def _get_objective_values(self) -> dict[str, float]:
        """Save the full array of objective values."""
        sol = self.solution
        objectives_values = self._wrapper_residuals(sol.x)
        objectives_values = {
            objective.name: objective_value
            for objective, objective_value in zip(
                self.objectives, objectives_values
            )
        }
        return objectives_values

    def _generate_optimisation_history(
        self,
        variables_values: np.ndarray,
        objectives_values: np.ndarray,
        constraints_values: np.ndarray,
    ) -> OptiInfo:
        """Create optimisation history."""
        opti_info = OptiInfo(
            hist_X=variables_values,
            hist_F=objectives_values,
            hist_G=constraints_values,
        )
        return opti_info
