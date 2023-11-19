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

import numpy as np

from optimisation.design_space.variable import Variable
from optimisation.design_space.constraint import Constraint
from optimisation.design_space.design_space import DesignSpace
from optimisation.design_space.helper import (same_value_as_nominal,
                                              phi_s_limits,
                                              phi_0_limits,
                                              k_e_limits,
                                              )

from core.list_of_elements.helper import equivalent_elt
from core.elements.element import Element

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput


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
    compensating_elements: list[Element]
    reference_elements: list[Element] | None = None

    def __post_init__(self):
        """Check that given elements can be retuned."""
        assert all([elt.can_be_retuned for elt in self.compensating_elements])

        self.filepath_variables: str
        self.filepath_constraints: str

    @property
    @abstractmethod
    def variables_names(self) -> tuple[str, ...]:
        """Return the name of the variables."""
        pass

    @property
    def constraints_names(self) -> tuple[str, ...]:
        """Return the name of the constraints."""
        return ()

    def use_files(self,
                  filepath_variables: str,
                  filepath_constraints: str | None = None) -> None:
        """Tell factory to generate design space from the provided files.

        Parameters
        ----------
        filepath_variables : str
            Path to the ``variables.csv`` file.
        filepath_constraints : str | None
            Path to the ``constraints.csv`` file. The default is None.

        """
        self.filepath_variables = filepath_variables
        if filepath_constraints is not None:
            self.filepath_constraints = filepath_constraints
        self.run = self._run_from_file

    def _run_variables(self) -> list[Variable]:
        """Set up all the required variables from presets."""
        assert self.reference_elements is not None
        variables = []
        for var_name in self.variables_names:
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    limits=self._get_limits_from_preset(var_name, ref_elt),
                    x_0=self._get_initial_value_from_preset(var_name, ref_elt),
                )
                variables.append(variable)
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Set up all the required constraints from presets."""
        assert self.reference_elements is not None
        constraints = []
        for constraint_name in self.constraints_names:
            for element in self.compensating_elements:
                ref_elt = equivalent_elt(self.reference_elements, element)
                constraint = Constraint(
                    name=constraint_name,
                    element_name=str(element),
                    limits=self._get_limits_from_preset(constraint_name,
                                                        ref_elt),
                )
                constraints.append(constraint)
        return constraints

    def run(self) -> DesignSpace:
        """Set up variables and constraints."""
        variables = self._run_variables()
        constraints = self._run_constraints()
        design_space = DesignSpace(variables, constraints)
        logging.info(str(design_space))
        return design_space

    def _get_initial_value_from_preset(self,
                                       variable: str,
                                       reference_element: Element) -> float:
        """Select initial value for given preset and parameter.

        The default behavior is to return the value of ``variable`` from
        ``reference_element``, which is a good starting point for optimisation.

        Parameters
        ----------
        variable : {'k_e', 'phi_0_rel', 'phi_0_abs', 'phi_s'}
            The variable from which you want the limits.
        reference_element : Element
            The element in its nominal tuning.

        Returns
        -------
        float
            Initial value.

        """
        return same_value_as_nominal(variable, reference_element)

    def _get_limits_from_preset(self,
                                variable: str,
                                reference_element: Element | None = None,
                                preset: str | None = None,
                                **kwargs) -> tuple[float, float]:
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
        assert self.reference_elements is not None, "Need reference_elements" \
            " to generate the design space with this preset."
        return LIMITS_GETTERS[preset](
            variable,
            reference_element=reference_element,
            reference_elements=self.reference_elements)

    def _run_from_file(self,
                       variables_names: tuple[str, ...],
                       constraints_names: tuple[str, ...] | None = None,
                       **kwargs: str,
                       ) -> DesignSpace:
        """Use the :meth:`.DesignSpace.from_files` constructor.

        Parameters
        ----------
        variables_names : tuple[str, ...]
            Name of the variables to create.
        constraints_names : tuple[str, ...] | None, optional
            Name of the constraints to create. The default is None.

        Returns
        -------
        DesignSpace

        """
        assert 'filepath_variables' in self.__dir__()
        filepath_constraints = getattr(self, 'filepath_constraints', None)

        elements_names = tuple([str(elt)
                                for elt in self.compensating_elements])
        design_space = DesignSpace.from_files(elements_names,
                                              self.filepath_variables,
                                              variables_names,
                                              filepath_constraints,
                                              constraints_names,
                                              **kwargs,
                                              )
        return design_space


@dataclass
class Unconstrained(DesignSpaceFactory):
    """Factory to set amplitude and phase of elements, no phi_s."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')


@dataclass
class ConstrainedSyncPhase(DesignSpaceFactory):
    """Factory to set k_e and phase of elements, with phi_s constraint."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')
    constraints_names: tuple[str] = ('phi_s', )


@dataclass
class SyncPhaseAsVariable(DesignSpaceFactory):
    """Factory to set k_e and phi_s of elements, no constraint."""

    variables_names: tuple[str, str] = ('phi_s', 'k_e')


@dataclass
class FM4_MYRRHA(DesignSpaceFactory):
    """Factory to set reduce design space around a pre-existing solution."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')

    def __post_init__(self):
        """Check that we are in the proper case."""
        super().__post_init__()
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

        for var_name in self.variables_names:
            for element in self.compensating_elements:
                my_initial_value = my_initial_values[var_name][str(element)]
                variable = Variable(
                    name=var_name,
                    element_name=str(element),
                    x_0=my_initial_value,
                    limits=(my_initial_value - tol, my_initial_value + tol),
                )
                variables.append(variable)
        return variables


@dataclass
class OneCavityMegaPower(DesignSpaceFactory):
    """Factory to have a element with huge power margins."""

    variables_names: tuple[str, str] = ('phi_0_abs', 'k_e')
    constraints_names: tuple[str] = ('phi_s', )

    def __post_init__(self) -> None:
        """Check that we are in the proper case."""
        super().__post_init__()
        assert len(self.compensating_elements) == 1, \
            "This case is designed to have ONE compensating elements (but " \
            "with huge power margins, so that it can compensate anything)."

    def _run_variables(self) -> list[Variable]:
        """Return normal variables, except very high k_e."""
        variables = []
        for var_name in self.variables_names:
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
        return variables

    def _run_constraints(self) -> list[Constraint]:
        """Return constraint on synchronous phase."""
        constraints = []
        for constraint_name in self.constraints_names:
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
    """
    Instantiante design space factory and create design space.

    .. todo::
        becoming less and less useful

    """
    assert isinstance(design_space_preset, str)
    design_space_factory_class = _read_design_space(design_space_preset)

    design_space_factory = design_space_factory_class(
        linac_name,
        compensating_elements,
        reference_elements,
    )

    design_space = design_space_factory.run()
    variables, constraints = design_space.variables, design_space.constraints
    compute_constraints = design_space.compute_constraints
    return variables, constraints, compute_constraints


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


myrrha_design_space = {
    'max_increase_sync_phase_in_percent': 40.,
    'max_absolute_sync_phase_in_rad': 0.,
    'min_absolute_sync_phase_in_rad': -np.pi / 2.,
    'max_decrease_k_e_in_percent': 50.,
    'max_increase_k_e_in_percent': 30.,
    'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section': True,
}

jaea_design_space = {
    'max_increase_sync_phase_in_percent': 50.,
    'max_absolute_sync_phase_in_rad': 0.,
    'min_absolute_sync_phase_in_rad': -np.pi / 2.,
    'max_decrease_k_e_in_percent': 50.,
    'max_increase_k_e_in_percent': 20.,
    'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section': False,
}

spiral2_design_space = {
    'max_increase_sync_phase_in_percent': 50.,
    'max_absolute_sync_phase_in_rad': 0.,
    'min_absolute_sync_phase_in_rad': -np.pi / 2.,
    'max_decrease_k_e_in_percent': 70.,
    'max_increase_k_e_in_percent': 20.,
    'maximum_k_e_is_calculated_wrt_maximum_k_e_of_section': False,
}
