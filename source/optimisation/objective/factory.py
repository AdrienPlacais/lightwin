"""Define a factory to create :class:`.Objective` objects.

When you implement a new objective preset, also add it to the list of
implemented presets in :mod:`config.optimisation.objective`.

.. todo::
    decorator to auto output the variables and constraints?

.. todo::
    Clarify that ``objective_position_preset`` should be understandable by
    :mod:`failures.position`.

"""

import logging
from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.helper import equivalent_elt
from core.list_of_elements.list_of_elements import ListOfElements
from experimental.test import assert_are_field_maps
from optimisation.design_space.helper import phi_s_limits
from optimisation.objective.minimize_difference_with_ref import (
    MinimizeDifferenceWithRef,
)
from optimisation.objective.minimize_mismatch import MinimizeMismatch
from optimisation.objective.objective import Objective
from optimisation.objective.position import zone_to_recompute
from optimisation.objective.quantity_is_between import QuantityIsBetween
from util.dicts_output import markdown


# =============================================================================
# Factories / presets
# =============================================================================
@dataclass
class ObjectiveFactory(ABC):
    """A base class to create :class:`Objective`.

    It is intended to be sub-classed to make presets. Look at
    :class:`EnergyPhaseMismatch` or :class:`EnergySyncPhaseMismatch` for
    examples.

    Attributes
    ----------
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
    design_space_kw : dict[str, str | bool | Path | float]
        Holds information on variables/constraints limits/initial values. Used
        to compute the limits that ``phi_s`` must respect when the synchronous
        phase is defined as an objective.

    """

    reference_elts: ListOfElements
    reference_simulation_output: SimulationOutput

    broken_elts: ListOfElements
    failed_elements: list[Element]
    compensating_elements: list[Element]

    design_space_kw: dict[str, str | bool | Path | float]

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
        """Determine which (sub)list of elements should be recomputed.

        You can override this method for your specific preset.

        """
        fault_idx = [
            element.idx["elt_idx"] for element in self.failed_elements
        ]
        comp_idx = [
            element.idx["elt_idx"] for element in self.compensating_elements
        ]

        if "position" in wtf:
            logging.warning(
                "position key should not be present in the .toml config file "
                "anymore. Its role is now fulfilled by the objective preset."
            )

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
        logging.info("\n".join(info))


class EnergyMismatch(ObjectiveFactory):
    """A set of two objectives: energy and mismatch.

    We try to match the kinetic energy and the mismatch factor at the end of
    the last altered lattice (the last lattice with a compensating or broken
    cavity).

    This set of objectives is adapted when you do not need to retrieve the
    absolute beam phase at the exit of the compensation zone, ie when rephasing
    all downstream cavities is not an issue.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set objective evaluation at end of last altered lattice."""
        objective_position_preset = ["end of last altered lattice"]
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            "full_lattices": False,
            "full_linac": False,
            "start_at_beginning_of_linac": False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensation_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[0]
        objectives = [
            self._get_w_kin(elt=last_element),
            self._get_mismatch(elt=last_element),
        ]
        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["w_kin"],
            weight=1.0,
            get_key="w_kin",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.0,
            get_key="twiss",
            get_kwargs={
                "elt": elt,
                "pos": "out",
                "to_numpy": True,
                "phase_space_name": "zdelta",
            },
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane.""",
        )
        return objective


class EnergyPhaseMismatch(ObjectiveFactory):
    """A set of three objectives: energy, absolute phase, mismatch.

    We try to match the kinetic energy, the absolute phase and the mismatch
    factor at the end of the last altered lattice (the last lattice with a
    compensating or broken cavity).
    With this preset, it is recommended to set constraints on the synchrous
    phase to help the optimisation algorithm to converge.

    This set of objectives is robust and rapid for ADS.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set objective evaluation at end of last altered lattice."""
        objective_position_preset = ["end of last altered lattice"]
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            "full_lattices": False,
            "full_linac": False,
            "start_at_beginning_of_linac": False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensation_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[0]
        objectives = [
            self._get_w_kin(elt=last_element),
            self._get_phi_abs(elt=last_element),
            self._get_mismatch(elt=last_element),
        ]
        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["w_kin"],
            weight=1.0,
            get_key="w_kin",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_phi_abs(self, elt: Element) -> Objective:
        """Return object to match phase."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["phi_abs"].replace("deg", "rad"),
            weight=1.0,
            get_key="phi_abs",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of phi_abs between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.0,
            get_key="twiss",
            get_kwargs={
                "elt": elt,
                "pos": "out",
                "to_numpy": True,
                "phase_space_name": "zdelta",
            },
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane.""",
        )
        return objective


class EnergySyncPhaseMismatch(ObjectiveFactory):
    """Match the synchronous phase, the energy and the mismatch factor.

    It is very similar to :class:`EnergyPhaseMismatch`, except that synchronous
    phases are declared as objectives.
    Objective will be 0 when synchronous phase is within the imposed limits.

    .. note::
        Do not set synchronous phases as constraints when using this preset.

    This set of objectives is slower than :class:`.EnergyPhaseMismatch`.
    However, it can help keeping the acceptance as high as possible.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set objective evaluation at end of last altered lattice."""
        objective_position_preset = ["end of last altered lattice"]
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            "full_lattices": False,
            "full_linac": False,
            "start_at_beginning_of_linac": False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        return [self.elts_of_compensation_zone[-1]]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy, phase and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[0]
        objectives = [
            self._get_w_kin(elt=last_element),
            self._get_phi_abs(elt=last_element),
            self._get_mismatch(elt=last_element),
        ]

        working_and_tunable_elements_in_compensation_zone = list(
            filter(
                lambda element: (
                    element.can_be_retuned
                    and element not in self.failed_elements
                ),
                self.elts_of_compensation_zone,
            )
        )

        assert_are_field_maps(
            working_and_tunable_elements_in_compensation_zone,
            detail="accessing phi_s property of a non field map",
        )

        objectives += [
            self._get_phi_s(element)
            for element in working_and_tunable_elements_in_compensation_zone
        ]

        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["w_kin"],
            weight=1.0,
            get_key="w_kin",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_phi_abs(self, elt: Element) -> Objective:
        """Return object to match phase."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["phi_abs"].replace("deg", "rad"),
            weight=1.0,
            get_key="phi_abs",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of phi_abs between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.0,
            get_key="twiss",
            get_kwargs={
                "elt": elt,
                "pos": "out",
                "to_numpy": True,
                "phase_space_name": "zdelta",
            },
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane.""",
        )
        return objective

    def _get_phi_s(self, cavity: FieldMap) -> Objective:
        """
        Objective to have sync phase within bounds.

        .. todo::
            Allow ``from_file``.

        """
        reference_cavity = equivalent_elt(self.reference_elts, cavity)

        if self.design_space_kw["from_file"]:
            raise IOError(
                "For now, synchronous phase cannot be taken from "
                "the variables or constraints.csv files when used as"
                " objectives."
            )
        limits = phi_s_limits(reference_cavity, **self.design_space_kw)

        objective = QuantityIsBetween(
            name=markdown["phi_s"].replace("deg", "rad"),
            weight=50.0,
            get_key="phi_s",
            get_kwargs={"elt": cavity, "pos": "out", "to_numpy": False},
            limits=limits,
            descriptor="""Synchronous phase should be between limits.""",
        )
        return objective


class EnergySeveralMismatches(ObjectiveFactory):
    """Match energy and mismatch (the latter on several periods).

    Experimental.

    """

    @property
    def objective_position_preset(self) -> list[str]:
        """Set where objective are evaluated."""
        objective_position_preset = [
            "end of last altered lattice",
            "one lattice after last altered lattice",
        ]
        return objective_position_preset

    @property
    def compensation_zone_override_settings(self) -> dict[str, bool]:
        """Set no particular overridings."""
        compensation_zone_override_settings = {
            "full_lattices": False,
            "full_linac": False,
            "start_at_beginning_of_linac": False,
        }
        return compensation_zone_override_settings

    def _elements_where_objective_are_evaluated(self) -> list[Element]:
        """Give element at end of compensation zone."""
        last_element = self.elts_of_compensation_zone[-1]
        elements_per_lattice = last_element.idx["lattice"]
        return [
            self.elts_of_compensation_zone[-1 - elements_per_lattice],
            last_element,
        ]

    def get_objectives(self) -> list[Objective]:
        """Give objects to match kinetic energy and mismatch factor."""
        last_element = self._elements_where_objective_are_evaluated()[-1]
        one_lattice_before = self._elements_where_objective_are_evaluated()[0]
        objectives = [
            self._get_w_kin(elt=one_lattice_before),
            self._get_mismatch(elt=one_lattice_before),
            self._get_mismatch(elt=last_element),
        ]
        self._output_objectives(objectives)
        return objectives

    def _get_w_kin(self, elt: Element) -> Objective:
        """Return object to match energy."""
        objective = MinimizeDifferenceWithRef(
            name=markdown["w_kin"],
            weight=1.0,
            get_key="w_kin",
            get_kwargs={"elt": elt, "pos": "out", "to_numpy": False},
            reference=self.reference_simulation_output,
            descriptor="""Minimize diff. of w_kin between ref and fix at the
            end of the compensation zone.
            """,
        )
        return objective

    def _get_mismatch(self, elt: Element) -> Objective:
        """Return object to keep mismatch as low as possible."""
        objective = MinimizeMismatch(
            name=r"$M_{z\delta}$",
            weight=1.0,
            get_key="twiss",
            get_kwargs={
                "elt": elt,
                "pos": "out",
                "to_numpy": True,
                "phase_space_name": "zdelta",
            },
            reference=self.reference_simulation_output,
            descriptor="""Minimize mismatch factor in the [z-delta] plane.""",
        )
        return objective


# =============================================================================
# Interface with LightWin
# =============================================================================
OBJECTIVE_PRESETS = {
    "EnergyPhaseMismatch": EnergyPhaseMismatch,
    "simple_ADS": EnergyPhaseMismatch,
    "EnergyMismatch": EnergyMismatch,
    "rephased_ADS": EnergyMismatch,
    "EnergySyncPhaseMismatch": EnergySyncPhaseMismatch,
    "sync_phase_as_objective_ADS": EnergySyncPhaseMismatch,
    "experimental": EnergySeveralMismatches,
}


def get_objectives_and_residuals_function(
    objective_preset: str,
    reference_elts: ListOfElements,
    reference_simulation_output: SimulationOutput,
    broken_elts: ListOfElements,
    failed_elements: list[Element],
    compensating_elements: list[Element],
    design_space_kw: dict[str, float | bool | str | Path],
) -> tuple[
    list[Element], list[Objective], Callable[[SimulationOutput], np.ndarray]
]:
    """Instantiate objective factory and create objectives.

    Parameters
    ----------
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
    design_space_kw : dict | None, optional
        Used when we need to determine the limits for ``phi_s``. Those limits
        are defined in the ``.ini`` configuration file.

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
    objective_factory_class = OBJECTIVE_PRESETS[objective_preset]

    objective_factory = objective_factory_class(
        reference_elts=reference_elts,
        reference_simulation_output=reference_simulation_output,
        broken_elts=broken_elts,
        failed_elements=failed_elements,
        compensating_elements=compensating_elements,
        design_space_kw=design_space_kw,
    )

    elts_of_compensation_zone = objective_factory.elts_of_compensation_zone
    objectives = objective_factory.get_objectives()
    compute_residuals = partial(_compute_residuals, objectives=objectives)
    return elts_of_compensation_zone, objectives, compute_residuals


def _compute_residuals(
    simulation_output: SimulationOutput, objectives: Collection[Objective]
) -> np.ndarray:
    """Compute residuals on given `Objectives` for given `SimulationOutput`."""
    residuals = [
        objective.evaluate(simulation_output) for objective in objectives
    ]
    return np.array(residuals)
