#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds the factory as well as the presets to handle objectives.

When you implement a new objective preset, also add it to the list of
implemented presets in :mod:`config.optimisation.objective`.

.. todo::
    decorator to auto output the variables and constraints?

.. todo::
    Clarify that ``objective_position_preset`` should be understandable by
    :mod:`failures.position`.

"""
import logging
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from typing import Callable, Any, Sequence
from functools import partial

import numpy as np

from optimisation.objective.objective import Objective
from optimisation.objective.minimize_difference_with_ref import \
    MinimizeDifferenceWithRef
from optimisation.objective.minimize_mismatch import MinimizeMismatch
from optimisation.objective.quantity_is_between import QuantityIsBetween
from optimisation.objective.position import zone_to_recompute

from optimisation.design_space.factory import LIMITS_GETTERS

from core.elements.element import Element
from core.elements.field_map import FieldMap

from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import equivalent_elt

from beam_calculation.output import SimulationOutput

from util.dicts_output import markdown
from experimental.test import assert_are_field_maps


# =============================================================================
# Factories / presets
# =============================================================================
@dataclass
class ObjectiveFactory(ABC):
    """
    A base class to create :class:`Objective`.

    It is intended to be sub-classed to make presets. Look at
    :class:`SimpleADS` or :class:`SyncPhaseAsObjectiveADS` for examples.

    Attributes
    ----------
    linac_name : str
        Name of the linac.
    reference_elts : ListOfElements
        All the reference elements.
    reference_simulation_output : SimulationOutput
        The reference simulation of the reference linac.
    broken_elts : ListOfElements
        List containing all the elements of the broken linac.
    failed_elements : list[Element]
        Cavities that failed.
    compensating_elements : list[Element]
        Cavities that will be used for the compensation.

    """

    linac_name: str
    reference_elts: ListOfElements
    reference_simulation_output: SimulationOutput

    broken_elts: ListOfElements
    failed_elements: list[Element]
    compensating_elements: list[Element]

    def __post_init__(self):
        """Determine the compensation zone."""
        assert all([elt.can_be_retuned for elt in self.compensating_elements])
        self.elts_of_compensation_zone = self._set_zone_to_recompute()

    @abstractmethod
    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Determine where objectives will be evaluated."""

    @abstractmethod
    def get_objectives(self) -> list[Objective]:
        """Create the :class:`Objective` instances."""

    @property
    @abstractmethod
    def objective_position_preset(self) -> list[str]:
        """
        Give a preset for :func:`failures.position.zone_to_recompute`.

        The returned values must be in the ``POSITION_TO_INDEX`` dictionary of
        :mod:`failures.position`.

        """
        pass

    @property
    @abstractmethod
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """
        Give flags for :func:`failures.position.zone_to_recompute`.

        The returned dictionary may have three flags:
            - full_lattices
            - full_linac
            - start_at_beginning_of_linac

        """
        pass

    def _set_zone_to_recompute(self, **wtf: Any) -> Sequence[Element]:
        """
        Determine which (sub)list of elements should be recomputed.

        You can override this method for your specific preset.

        """
        fault_idx = [element.idx['elt_idx']
                     for element in self.failed_elements]
        comp_idx = [element.idx['elt_idx']
                    for element in self.compensating_elements]

        if 'position' in wtf:
            logging.warning("position key should not be present in the .ini "
                            "file anymore. Its role is now fulfilled by the "
                            "objective preset.")

        elts_of_compensation_zone = zone_to_recompute(
            self.broken_elts,
            self.objective_position_preset,
            fault_idx,
            comp_idx,
            **self.compensation_zone_override_settings,
        )
        return elts_of_compensation_zone

    @staticmethod
    def _output_objectives(objectives: list[Objective]) -> None:
        """Print information on the objectives that were created."""
        info = [str(objective) for objective in objectives]
        info.insert(0, "Created objectives:")
        info.insert(1, "=" * 100)
        info.insert(2, Objective.str_header())
        info.insert(3, "-" * 100)
        info.append("=" * 100)
        logging.info('\n'.join(info))


class SimpleADS(ObjectiveFactory):
    """
    A rapid and relatively robust method for ADS.

    We try to match the kinetic energy, the absolute phase and the mismatch
    factor at the end of the last altered lattice (the last lattice with a
    compensating or broken cavity).
    With this preset, it is recommended to set constraints on the synchrous
    phase to help the optimisation algorithm to converge.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set objective evaluation at end of last altered lattice."""
        objective_position_preset = ['end of last altered lattice']
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            'full_lattices': False,
            'full_linac': False,
            'start_at_beginning_of_linac': False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensation_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[0]
        objectives = [self._get_w_kin(elt=last_element),
                      self._get_phi_abs(elt=last_element),
                      self._get_mismatch(elt=last_element)]
        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown['w_kin'],
            weight=1.,
            get_key='w_kin',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """
        )
        return objective

    def _get_phi_abs(self, elt: Element) -> Objective:
        """Return object to match phase."""
        objective = MinimizeDifferenceWithRef(
            name=markdown['phi_abs'].replace('deg', 'rad'),
            weight=1.,
            get_key='phi_abs',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of phi_abs between ref and fix at the
            end of the compensation zone.
            """
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.,
            get_key='twiss',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': True,
                        'phase_space': 'zdelta'},
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane."""
        )
        return objective


class SyncPhaseAsObjectiveADS(ObjectiveFactory):
    """
    Factory to handle synchronous phases as objectives.

    It is very similar to :class:`SimpleADS`, except that synchronous phases
    are declared as objectives.
    Objective will be 0 when synchronous phase is within the imposed limits.

    .. note::
        Do not set synchronous phases as constraints when using this preset.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set objective evaluation at end of last altered lattice."""
        objective_position_preset = ['end of last altered lattice']
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            'full_lattices': False,
            'full_linac': False,
            'start_at_beginning_of_linac': False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensation_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[0]
        objectives = [self._get_w_kin(elt=last_element),
                      self._get_phi_abs(elt=last_element),
                      self._get_mismatch(elt=last_element)]

        working_and_tunable_elements_in_compensation_zone = list(filter(
            lambda element: (element.can_be_retuned
                             and element not in self.failed_elements),
            self.elts_of_compensation_zone))

        assert_are_field_maps(
            working_and_tunable_elements_in_compensation_zone,
            detail='accessing phi_s property of a non field map')

        objectives += [self._get_phi_s(element)
                       for element in
                       working_and_tunable_elements_in_compensation_zone]

        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown['w_kin'],
            weight=1.,
            get_key='w_kin',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """
        )
        return objective

    def _get_phi_abs(self, elt: Element) -> Objective:
        """Return object to match phase."""
        objective = MinimizeDifferenceWithRef(
            name=markdown['phi_abs'].replace('deg', 'rad'),
            weight=1.,
            get_key='phi_abs',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of phi_abs between ref and fix at the
            end of the compensation zone.
            """
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.,
            get_key='twiss',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': True,
                        'phase_space': 'zdelta'},
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane."""
        )
        return objective

    def _get_phi_s(self, cavity: FieldMap) -> Objective:
        """Objective to have sync phase within bounds."""
        reference_cavity = equivalent_elt(self.reference_elts, cavity)
        limits_getter = LIMITS_GETTERS[self.linac_name]
        limits = limits_getter('phi_s', reference_cavity)

        objective = QuantityIsBetween(
            name=markdown['phi_s'].replace('deg', 'rad'),
            weight=50.,
            get_key='phi_s',
            get_kwargs={'elt': cavity, 'pos': 'out', 'to_numpy': False},
            limits=limits,
            descriptor="""Synchronous phase should be between limits."""
        )
        return objective


def _read_objective(objective_preset: str) -> ABCMeta:
    """Return proper factory."""
    factories = {
        'simple_ADS': SimpleADS,
        'sync_phase_as_objective_ADS': SyncPhaseAsObjectiveADS,
    }
    return factories[objective_preset]


# =============================================================================
# Interface with LightWin
# =============================================================================
def get_objectives_and_residuals_function(
    linac_name: str,
    objective_preset: str,
    reference_elts: ListOfElements,
    reference_simulation_output: SimulationOutput,
    broken_elts: ListOfElements,
    failed_elements: list[Element],
    compensating_elements: list[Element],
) -> tuple[list[Element],
           list[Objective],
           Callable[[SimulationOutput], np.ndarray]]:
    """
    Instantiate objective factory and create objectives.

    Parameters
    ----------
    linac_name : str
        Name of the linac.
    reference_elts : ListOfElements
        All the reference elements.
    reference_simulation_output : SimulationOutput
        The reference simulation of the reference linac.
    broken_elts : ListOfElements
        The elements of the broken linac.
    failed_elements : list[Element]
        Elements that failed.
    compensating_elements : list[Element]
        Elements that will be used for the compensation.

    Returns
    -------
    elts_of_compensation_zone : list[Element]
        Portion of the linac that will be recomputed during the optimisation
        process.
    objectives : list[Objective]
        Objectives that the optimisation algorithm will try to match.
    compute_residuals : Callable[[SimulationOutput], np.ndarray]
        Function that converts a :class:`.SimulationOutput` to a plain numpy
        array of residues.

    """
    assert isinstance(objective_preset, str)
    objective_factory_class = _read_objective(objective_preset)

    objective_factory = objective_factory_class(
        linac_name=linac_name,
        reference_elts=reference_elts,
        reference_simulation_output=reference_simulation_output,
        broken_elts=broken_elts,
        failed_elements=failed_elements,
        compensating_elements=compensating_elements,
    )

    elts_of_compensation_zone = objective_factory.elts_of_compensation_zone
    objectives = objective_factory.get_objectives()
    compute_residuals = partial(_compute_residuals, objectives=objectives)
    return elts_of_compensation_zone, objectives, compute_residuals


def _compute_residuals(simulation_output: SimulationOutput,
                       objectives: list[Objective]) -> np.ndarray:
    """Compute residuals on given `Objectives` for given `SimulationOutput`."""
    residuals = [objective.evaluate(simulation_output)
                 for objective in objectives]
    return np.array(residuals)
