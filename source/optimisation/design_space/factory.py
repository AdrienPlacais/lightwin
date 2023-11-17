#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define factory and presets to handle variables, constraints, limits, etc..

When you add you own presets, do not forget to add them to the list of
implemented presets in :mod:`config.optimisation.design_space`.

.. todo::
    decorator to auto output the variables and constraints?

.. todo::
    ``pyright`` is not very happy about this module.

.. todo::
    initial values, limits, etc would be better in every linac own project.

"""
import logging
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Callable, Any, Sequence
from functools import partial

import numpy as np

from optimisation.design_space.variable import Variable
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.design_space import DesignSpace

from core.list_of_elements.helper import equivalent_elt
from core.elements.element import Element

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from util.helper import chunks, flatten


# =============================================================================
# Factories / presets
# =============================================================================
@dataclass
class DesignSpaceFactory(ABC):
    """
    A base class to handle :class:`Variable` and :class:`Constraint` creation.

    Attributes
    ----------
    preset : str
        The name of the linac, to select the limits (in particular for ``k_e``
        and ``phi_s``) and the initial values.
    reference_elements : list[Element]
       All the elements with the reference setting.
    compensating_elements : list[Element]
        The elements from the linac under fixing that will be used for
        compensation.

    """

    preset: str
    reference_elements: list[Element]
    compensating_elements: list[Element]

    def __post_init__(self):
        """Check that given elements can be retuned."""
        assert all([elt.can_be_retuned for elt in self.compensating_elements])

    @abstractmethod
    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables."""

    @abstractmethod
    def _run_constraints(self) -> list[Constraint]:
        """Set up all the required constraints."""

    def run(self) -> DesignSpace:
        """Set up variables and constraints."""
        variables = self._run_variables()
        constraints = self._run_constraints()
        design_space = DesignSpace(variables, constraints)
        return design_space

    def _get_initial_value_from_preset(self,
                                       variable: str,
                                       reference_element: Element,
                                       preset: str | None = None,
                                       **kwargs) -> float:
        """Select initial value for given preset and parameter.

        Call this method for classic initial values.

        Parameters
        ----------
        variable : {'k_e', 'phi_0_rel', 'phi_0_abs', 'phi_s'}
            The variable from which you want the limits.
        reference_element : Element | None, optional
            The element in its nominal tuning. The default is None.
        preset : str | None, optional
            Key of the ``INITIAL_VALUE_GETTERS`` dict to select proper initial
            value. The default is None, in which case we take ``self.preset``.

        Returns
        -------
        float
            Initial value.

        """
        if preset is None:
            preset = self.preset
        return INITIAL_VALUE_GETTERS[preset](
            variable,
            reference_element=reference_element)

    def _get_limits_from_preset(self,
                                variable: str,
                                reference_element: Element | None = None,
                                preset: str | None = None,
                                **kwargs) -> tuple[float | None]:
        """
        Select limits for given preset and parameter.

        Call this method for classic limits.

        Parameters
        ----------
        variable : {'k_e', 'phi_0_rel', 'phi_0_abs', 'phi_s'}
            The variable from which you want the limits.
        reference_element : Element | None, optional
            The element in its nominal tuning. The default is None.
        preset : str | None, optional
            Key of the ``LIMITS_GETTERS`` dict to select proper initial value.
            The default is None, in which case we take ``self.preset``.

        Returns
        -------
        tuple[float | None]
            Lower and upper limit for current variable.

        """
        if preset is None:
            preset = self.preset
        return LIMITS_GETTERS[preset](
            variable,
            reference_element=reference_element,
            reference_elements=self.reference_elements)

    @staticmethod
    def _output_variables(variables: list[Variable]) -> None:
        """Print information on the variables that were created."""
        info = [str(variable) for variable in variables]
        info.insert(0, "Created variables:")
        info.insert(1, "=" * 100)
        info.insert(2, Variable.header_of__str__())
        info.insert(3, "-" * 100)
        info.append("=" * 100)
        logging.info('\n'.join(info))

    @staticmethod
    def _output_constraints(constraints: list[Constraint]) -> None:
        """Print information on the constraints that were created."""
        info = [str(constraint) for constraint in constraints]
        info.insert(0, "Created constraints:\n")
        info.insert(1, "=" * 100)
        info.insert(2, Constraint.header_of__str__())
        info.insert(3, "-" * 100)
        info.append("=" * 100)
        logging.info('\n'.join(info))


class Unconstrained(DesignSpaceFactory):
    """Factory to set amplitude and phase of elements, no phi_s."""

    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables."""
        variables = []
        for var_name in ('phi_0_abs', 'k_e'):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    limits=self._get_limits_from_preset(var_name, ref_elt),
                    x_0=self._get_initial_value_from_preset(var_name, ref_elt),
                )
                variables.append(variable)
        self._output_variables(variables)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return no constraint."""
        constraints = []
        self._output_constraints(constraints)
        return constraints


class ConstrainedSyncPhase(DesignSpaceFactory):
    """Factory to set k_e and phase of elements, with phi_s constraint."""

    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables."""
        variables = []
        for var_name in ('phi_0_abs', 'k_e'):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=self._get_initial_value_from_preset(var_name, ref_elt),
                    limits=self._get_limits_from_preset(var_name, ref_elt),
                )
                variables.append(variable)
        self._output_variables(variables)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return constraint on synchronous phase."""
        constraints = []
        for constraint_name in ('phi_s',):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                constraint = Constraint(
                    name=constraint_name,
                    element_name=str(element),
                    limits=self._get_limits_from_preset(constraint_name,
                                                        ref_elt),
                )
                constraints.append(constraint)
        self._output_constraints(constraints)
        return constraints


class SyncPhaseAsVariable(DesignSpaceFactory):
    """Factory to set k_e and phi_s of elements, no constraint."""

    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables."""
        variables = []
        for var_name in ('phi_s', 'k_e'):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=self._get_initial_value_from_preset(var_name, ref_elt),
                    limits=self._get_limits_from_preset(var_name, ref_elt),
                )
                variables.append(variable)
        self._output_variables(variables)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return no constraint."""
        constraints = []
        self._output_constraints(constraints)
        return constraints


class FM4_MYRRHA(DesignSpaceFactory):
    """Factory to set reduce design space around a pre-existing solution."""

    def __post_init__(self):
        """Check that we are in the proper case."""
        assert self.preset == 'MYRRHA'
        assert [str(elt) for elt in self.compensating_elements] == [
            'FM1', 'FM2', 'FM3', 'FM5', 'FM6']

    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables."""
        variables = []
        my_initial_values = {'phi_0_abs': {'FM1': 1.2428429564125352,
                                           'FM2': 5.849758478693384,
                                           'FM3': 1.370628110261926,
                                           'FM5': 3.323382937071699,
                                           'FM6': 2.611163043271624
                                           },
                             'k_e': {'FM1': 1.614713,
                                     'FM2': 1.607485,
                                     'FM3': 1.9268,
                                     'FM5': 1.942578,
                                     'FM6': 1.851571,
                                     }
                             }
        tol = 1e-3

        for var_name in ('phi_0_abs', 'k_e'):
            for element in self.compensating_elements:
                my_initial_value = my_initial_values[var_name][str(element)]
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=my_initial_value,
                    limits=(my_initial_value - tol, my_initial_value + tol),
                )
                variables.append(variable)
        self._output_variables(variables)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return no constraint."""
        constraints = []
        self._output_constraints(constraints)
        return constraints


class OneCavityMegaPower(DesignSpaceFactory):
    """Factory to have a element with huge power margins."""

    def __post_init__(self) -> None:
        """Check that we are in the proper case."""
        assert len(self.compensating_elements) == 1, \
            "This case is designed to have ONE compensating elements (but " \
            "with huge power margins, so that it can compensate anything)."

    def _run_variables(self) -> list[Variable]:
        """Return normal variables, except very high k_e."""
        variables = []
        for var_name in ('phi_0_abs', 'k_e'):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)

                limits = self._get_limits_from_preset(var_name, ref_elt)
                if var_name == 'k_e':
                    limits = (limits[0], 10. * limits[1])

                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=self._get_initial_value_from_preset(var_name, ref_elt),
                    limits=limits,
                )
                variables.append(variable)
        self._output_variables(variables)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return constraint on synchronous phase."""
        constraints = []
        for constraint_name in ('phi_s',):
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                constraint = Constraint(
                    name=constraint_name,
                    element_name=str(element),
                    limits=self._get_limits_from_preset(constraint_name,
                                                        ref_elt,
                                                        preset='MYRRHA'),
                )
                constraints.append(constraint)
        self._output_constraints(constraints)
        return constraints


def _read_design_space(design_space_preset: str) -> ABCMeta:
    """Return proper factory."""
    factories = {
        'unconstrained': Unconstrained,
        'constrained_sync_phase': ConstrainedSyncPhase,
        'sync_phase_as_variable': SyncPhaseAsVariable,
        'FM4_MYRRHA': FM4_MYRRHA,
        'one_element_mega_power': OneCavityMegaPower,
    }
    return factories[design_space_preset]


# =============================================================================
# Interface with LightWin
# =============================================================================
def get_design_space_and_constraint_function(
    linac_name: str,
    design_space_preset: str,
    reference_elements: Sequence[Element],
    compensating_elements: list[Element],
    **wtf: Any) -> tuple[list[Variable],
                         list[Constraint],
                         Callable[[SimulationOutput], np.ndarray]]:
    """Instantiante design space factory and create design space."""
    assert isinstance(design_space_preset, str)
    design_space_factory_class = _read_design_space(design_space_preset)

    design_space_factory = design_space_factory_class(
        linac_name,
        compensating_elements,
        reference_elements,
    )

    design_space = design_space_factory.run()
    variables, constraints = design_space.variables, design_space.constraints
    compute_constraints = partial(_compute_constraints,
                                  constraints=constraints)
    return variables, constraints, compute_constraints


def _compute_constraints(simulation_output: SimulationOutput,
                         constraints: list[Constraint]) -> np.ndarray:
    """Compute constraint violation for given `SimulationOutput`."""
    constraints_with_tuples = [constraint.evaluate(simulation_output)
                               for constraint in constraints]
    constraint_violation = [
        single_constraint
        for constraint_with_tuples in constraints_with_tuples
        for single_constraint in constraint_with_tuples
        if ~np.isnan(single_constraint)
    ]
    return np.array(constraint_violation)


# =============================================================================
# Initial value for k_e, phi_0, phi_s for every implemented linac
# =============================================================================
def _initial_value_myrrha(variable: str,
                          reference_element: Element | None = None,
                          **kwargs) -> float | None:
    """Set the initial value for a quantity in MYRRHA ADS linac."""
    reference_value = reference_element.get(variable, to_numpy=False)
    myrrha_initial_value = {
        'phi_s': lambda reference_value: reference_value,
        'k_e': lambda reference_value: reference_value,
        'phi_0_rel': lambda reference_value: reference_value,
        'phi_0_abs': lambda reference_value: reference_value,
    }
    if variable not in myrrha_initial_value:
        logging.error(f"Preset MYRRHA has no preset for {variable}.")
        return None
    return myrrha_initial_value[variable](reference_value)


def _initial_value_jaea(variable: str,
                        reference_element: Element | None = None,
                        **kwargs) -> float:
    """Set the initial value for a quantity in JAEA ADS linac."""
    reference_value = reference_element.get(variable, to_numpy=False)
    jaea_initial_value = {
        'phi_s': lambda reference_value: reference_value,
        'k_e': lambda reference_value: reference_value,
        'phi_0_rel': lambda reference_value: reference_value,
        'phi_0_abs': lambda reference_value: reference_value,
    }
    if variable not in jaea_initial_value:
        logging.error(f"Preset JAEA has no preset for {variable}.")
        return None
    return jaea_initial_value[variable](reference_value)


def _initial_value_spiral2(variable: str,
                           reference_element: Element | None = None,
                           **kwargs) -> float:
    """Set the initial_value for a quantity in SPIRAL2 linac."""
    reference_value = reference_element.get(variable, to_numpy=False)
    spiral2_initial_value = {
        'phi_s': lambda reference_value: reference_value,
        'k_e': lambda reference_value: reference_value,
        'phi_0_rel': lambda reference_value: reference_value,
        'phi_0_abs': lambda reference_value: reference_value,
    }
    if variable not in spiral2_initial_value:
        logging.error(f"Preset SPIRAL2 has no preset for {variable}.")
        return None
    return spiral2_initial_value[variable](reference_value)


INITIAL_VALUE_GETTERS = {
    'MYRRHA': _initial_value_myrrha,
    'JAEA': _initial_value_jaea,
    'SPIRAL2': _initial_value_spiral2,
}


# =============================================================================
# Limits for k_e, phi_0, phi_s for every implemented linac
# =============================================================================
def _limits_myrrha(variable: str,
                   reference_element: Element | None = None,
                   reference_elements: list[Element] | None = None,
                   **kwargs) -> tuple[float | None]:
    """
    Set the limits for a quantity in MYRRHA ADS linac.

    Parameters
    ----------
    variable : {'k_e', 'phi_s', 'phi_0_abs', 'phi_0_rel'}
        Quantity under study.
    reference_element : Element | None, optional
        Cavity with nominal settings. The default is None.
    reference_elements : list[Element] | None, optional
        List holding all the reference elements, in their nominal settings. The
        default is None.

    Returns
    -------
    tuple[float | None]
        Lower and upper limit for the current ``variable`` and element. None
        means that there is no limit.

    """
    reference_value = reference_element.get(variable, to_numpy=False)
    myrrha_limits = {
        'phi_s': lambda reference_value: (
            # Minimum: -90deg
            -np.pi / 2.,
            # Maximum: 0deg or reference + 40%           (reminder: phi_s < 0)
            min(0., reference_value * (1. - 0.4))
        ),
        'k_e': lambda reference_value: (
            # Minimum: 50% of ref k_e
            reference_value * 0.5,
            # Maximum: maximum of section + 30%
            1.3 * _get_maximum_k_e_of_section(reference_element.idx['section'],
                                              reference_elements)
        ),
        'phi_0_rel': lambda reference_value: (-2. * np.pi, 2. * np.pi),
        'phi_0_abs': lambda reference_value: (-2. * np.pi, 2. * np.pi),
    }
    if variable not in myrrha_limits:
        logging.error(f"Preset MYRRHA has no preset for {variable}.")
        return (None, None)
    return myrrha_limits[variable](reference_value)


def _limits_jaea(variable: str,
                 reference_element: Element | None = None,
                 **kwargs) -> tuple[float | None]:
    """Set the limits for a quantity in JAEA ADS linac."""
    reference_value = reference_element.get(variable, to_numpy=False)
    jaea_limits = {
        'phi_s': lambda reference_value: (
            # Minimum: -90deg
            -np.pi / 2.,
            # Maximum: 0deg or reference + 50%           (reminder: phi_s < 0)
            min(0., reference_value * (1. - 0.5))
        ),
        'k_e': lambda reference_value: (0.5 * reference_value,
                                        1.2 * reference_value),
        'phi_0_rel': lambda reference_value: (-2. * np.pi, 2. * np.pi),
        'phi_0_abs': lambda reference_value: (-2. * np.pi, 2. * np.pi),
    }
    if variable not in jaea_limits:
        logging.error(f"Preset JAEA has no preset for {variable}.")
        return (None, None)
    return jaea_limits[variable](reference_value)


def _limits_spiral2(variable: str,
                    reference_element: Element | None = None,
                    **kwargs) -> tuple[float | None]:
    """Set the limits for a quantity in SPIRAL2 linac."""
    reference_value = reference_element.get(variable, to_numpy=False)
    spiral2_limits = {
        'phi_s': lambda reference_value: (
            # Minimum: -90deg
            -np.pi / 2.,
            # Maximum: 0deg or reference + 50%           (reminder: phi_s < 0)
            min(0., reference_value * (1. - 0.5))
        ),
        'k_e': lambda reference_value: (0.3 * reference_value,
                                        1.05 * reference_value),
        'phi_0_rel': lambda reference_value: (-2. * np.pi, 2. * np.pi),
        'phi_0_abs': lambda reference_value: (-2. * np.pi, 2. * np.pi),
    }
    if variable not in spiral2_limits:
        logging.error(f"Preset SPIRAL2 has no preset for {variable}.")
        return (None, None)
    return spiral2_limits[variable](reference_value)


def _get_maximum_k_e_of_section(section_idx: int,
                                reference_elements: list[Element],
                                ) -> float:
    """Get the maximum ``k_e`` of section."""
    elements_in_current_section = list(filter(
        lambda element: element.idx['section'] == section_idx,
        reference_elements))
    k_e_in_current_section = [element.get('k_e', to_numpy=False)
                              for element in elements_in_current_section]
    maximum_k_e = np.nanmax(k_e_in_current_section)
    return maximum_k_e


LIMITS_GETTERS = {
    'MYRRHA': _limits_myrrha,
    'JAEA': _limits_jaea,
    'SPIRAL2': _limits_spiral2,
}
