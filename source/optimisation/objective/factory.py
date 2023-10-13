#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:44:41 2023.

@author: placais

This module holds the factory as well as the presets to handle objectives.

.. todo::
    decorator to auto output the variables and constraints?

"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable
from functools import partial

import numpy as np

from optimisation.objective.objective import Objective
from optimisation.objective.minimize_difference_with_ref import \
    MinimizeDifferenceWithRef
from optimisation.objective.minimize_mismatch import MinimizeMismatch
from optimisation.objective.quantity_is_between import QuantityIsBetween

from optimisation.design_space.factory import LIMITS_GETTERS

from core.elements.element import Element
from core.elements.field_map import FieldMap

from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import equiv_elt

from beam_calculation.output import SimulationOutput

from util.dicts_output import markdown


# =============================================================================
# Factories / presets
# =============================================================================
@dataclass
class ObjectiveFactory(ABC):
    """
    A base class to handle :class:`Objective` objects creation.

    Attributes
    ----------
    linac_name : str
        Name of the linac.
    reference : SimulationOutput
        The reference simulation of the reference linac.
    elts_of_compensating_zone : list[Element]
        All the elements in the compensating zone.
    failed_cavities : list[FieldMap]
        Cavities that failed.
    reference_elts : ListOfElements
        All the reference elements.
    need_to_add_element_to_compensating_zone : bool
        True when the objectives should be checked outside of the compensating
        zone (e.g. one lattice after last compensating zone). In this case,
        ``elts_of_compensating_zone`` should be updated and returned to the
        rest of the code.

    """

    linac_name: str
    reference: SimulationOutput
    elts_of_compensating_zone: list[Element]
    failed_cavities: list[FieldMap]
    reference_elts: ListOfElements
    need_to_add_element_to_compensating_zone: bool

    def __post_init__(self) -> None:
        """Check validity of some inputs."""
        if self.need_to_add_element_to_compensating_zone:
            raise NotImplementedError("Current objective needs to be evaluated"
                                      " outside of the compensation zone. "
                                      "Hence it should be extended, which is "
                                      "currently not supported.")

    @abstractmethod
    def _get_positions(self) -> list[Element]:
        """Determine where objectives will be evaluated."""

    @abstractmethod
    def get_objectives(self) -> list[Objective]:
        """Create the :class:`Objective` instances."""

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
    """Factory aimed at efficient compensation for ADS linacs."""

    def _get_positions(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensating_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._get_positions()[0]
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
            reference=self.reference,
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
            reference=self.reference,
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
            reference=self.reference,
            descriptor="""Minimize mismatch factor in the [z-delta] plane."""
        )
        return objective


class SyncPhaseAsObjectiveADS(ObjectiveFactory):
    """Factory to handle synchronous phases."""

    def _get_positions(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensating_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._get_positions()[0]
        objectives = [self._get_w_kin(elt=last_element),
                      self._get_phi_abs(elt=last_element),
                      self._get_mismatch(elt=last_element)]

        working_cavities_in_compensating_zone = list(filter(
            lambda cavity: (isinstance(cavity, FieldMap)
                            and cavity not in self.failed_cavities),
            self.elts_of_compensating_zone))

        objectives += [self._get_phi_s(cavity)
                       for cavity in working_cavities_in_compensating_zone]

        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown['w_kin'],
            weight=1.,
            get_key='w_kin',
            get_kwargs={'elt': elt, 'pos': 'out', 'to_numpy': False},
            reference=self.reference,
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
            reference=self.reference,
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
            reference=self.reference,
            descriptor="""Minimize mismatch factor in the [z-delta] plane."""
        )
        return objective

    def _get_phi_s(self, cavity: FieldMap) -> Objective:
        """Objective to have sync phase within bounds."""
        reference_cavity = equiv_elt(self.reference_elts, cavity)
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


def _read_objective(objective_preset: str) -> ObjectiveFactory:
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
    reference_simulation_output: SimulationOutput,
    elts_of_compensating_zone: list[Element],
    failed_cavities: list[FieldMap],
    reference_elts: ListOfElements,
) -> tuple[list[Objective], Callable[[SimulationOutput], np.ndarray]]:
    """Instantiate objective factory and create objectives."""
    objective_factory = _read_objective(objective_preset)

    objective_factory_instance = objective_factory(
        linac_name,
        reference_simulation_output,
        elts_of_compensating_zone,
        failed_cavities,
        reference_elts,
        need_to_add_element_to_compensating_zone=False,
    )

    objectives = objective_factory_instance.get_objectives()
    compute_residuals = partial(_compute_residuals, objectives=objectives)

    return objectives, compute_residuals


def _compute_residuals(simulation_output: SimulationOutput,
                       objectives: list[Objective]) -> np.ndarray:
    """Compute residuals on given `Objectives` for given `SimulationOutput`."""
    residuals = [objective.evaluate(simulation_output)
                 for objective in objectives]
    return np.array(residuals)
