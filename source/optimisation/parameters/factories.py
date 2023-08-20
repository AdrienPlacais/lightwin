#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:44:41 2023.

@author: placais
"""
import logging
from typing import Callable
from functools import partial

import numpy as np

from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint
from optimisation.parameters.objective import Objective

from core.list_of_elements import ListOfElements, equiv_elt
from core.elements import FieldMap, _Element

from beam_calculation.output import SimulationOutput


# =============================================================================
# Factories
# =============================================================================
def variable_factory(preset: str,
                     variable_names: list[str],
                     compensating_cavities: list[FieldMap],
                     ref_elts: ListOfElements,
                     global_compensation: bool = False,
                     ) -> list[Variable]:
    """Create the necessary `Variable` objects."""
    variables = []

    for var_name in variable_names:
        initial_value_calculator = INITIAL_VALUE_CALCULATORS[var_name]
        limits_calculator = LIMITS_CALCULATORS[var_name]

        for cavity in compensating_cavities:
            ref_cav = equiv_elt(ref_elts, cavity)
            kwargs = {
                'preset': preset,
                'ref_cav': ref_cav,
                'ref_elts': ref_elts,
                'global_compensation': global_compensation,
            }
            variable = Variable(name=var_name,
                                cavity_name=str(cavity),
                                x_0=initial_value_calculator(ref_cav),
                                limits=limits_calculator(**kwargs)
                                )
            variables.append(variable)

    message = [str(variable) for variable in variables]
    message.insert(0, "Variables generated from presets in optimisation."
                   "parameters.factories:")
    logging.info('\n'.join(message))
    return variables


def constraint_factory(preset: str,
                       constraint_names: list[str],
                       compensating_cavities: list[FieldMap],
                       ref_elts: ListOfElements,
                       ) -> tuple[list[Constraint],
                                  Callable[SimulationOutput, np.ndarray]]:
    """
    Create the necessary `Constraint` objects.

    Parameters
    ----------
    preset : str
        Name of the linac, used to select proper phi_s policy.
    constraint_names : list[str]
        List of the names of the quantities under constraint.
    compensating_cavities : list[FieldMap]
        List of the compensating cavities in which constraint must be
        evaluated.
    ref_elts : ListOfElements
        Reference list of elements, with reference (nominal) phi_s in
        particular.

    Returns
    -------
    constraints : list[Constraint]
        List containing the `Constraint` objects.
    compute_constraints : Callable[SimulationOutput, np.ndarray]
        Compute the constraint violation for a given `SimulationOutput`.

    """
    constraints = []
    for var_name in constraint_names:
        limits_calculator = LIMITS_CALCULATORS[var_name]

        for cavity in compensating_cavities:
            kwargs = {
                'preset': preset,
                'ref_cav': equiv_elt(ref_elts, cavity),
                'ref_elts': ref_elts,
            }
            constraint = Constraint(name=var_name,
                                    cavity_name=str(cavity),
                                    limits=limits_calculator(**kwargs)
                                    )
            constraints.append(constraint)

    message = [str(constraint) for constraint in constraints]
    message.insert(0, "Constraints generated from presets in optimisation."
                   "parameters.factories:")
    logging.info('\n'.join(message))

    compute_constraints = partial(_compute_constraints,
                                  constraints=constraints)
    return constraints, compute_constraints


def objective_factory(names: list[str],
                      scales: list[float],
                      elements: list[_Element],
                      reference_simulation_output: SimulationOutput,
                      positions: list[str] | None = None,
                      ) -> tuple[list[Objective],
                                 Callable[SimulationOutput, np.ndarray]]:
    """Create the required `Objective` objects.

    Parameters
    ----------
    names : list[str]
        Name of the objectives. All individual objectives have to work with the
        `SimulationOutput.get` method.
    scales : list[float]
        List of the scales for every `_Element` and every objective. If you
        have 5 objectives, the 5 first `scale` will be applied to the 5
        objectives at the first `_Element`, the 5 following at the second
        `_Element`, etc.
    elements : list[_Element]
        List of `_Element` where objectives should be evaluated.
    reference_simulation_output : SimulationOutput
        A `SimulationOutput` on the reference linac (no fault).
    positions : list[float] | None, optional
        Where objectives should be evaluated for each `_Element`. The default
        is None, which comes back to evaluating objective at the exit of every
        `_Element`.

    Returns
    -------
    objectives : list[Objective]
        A list of the `Objective` objects.
    compute_residuals : Callable[SimulationOutput, np.ndarray]
        A function that takes in a `SimulationOutput` and returns the residues
        of every objective w.r.t the reference one.

    """
    objectives = []
    idx_scale = 0

    if positions is None:
        positions = ['out' for element in elements]

    for element, position in zip(elements, positions):
        for name in names:
            scale = scales[idx_scale]

            if scale == 0.:
                continue

            kwargs = {
                'name': name,
                'scale': scale,
                'element': element,
                'pos': position,
                'reference_simulation_output': reference_simulation_output
            }

            objective = Objective(**kwargs)
            objectives.append(objective)
            idx_scale += 1

    message = [objective.ref for objective in objectives]
    message.insert(0, "Objectives, scales, initial values:")
    logging.info('\n'.join(message))

    compute_residuals = partial(_compute_residuals, objectives=objectives)
    return objectives, compute_residuals


# =============================================================================
# Presets for variable limits
# =============================================================================
def _limits_k_e(preset: str | None = None,
                ref_cav: FieldMap | None = None,
                ref_elts: ListOfElements | None = None,
                global_compensation: bool = False,
                **kwargs
                ) -> tuple[float | None]:
    """Limits for electric field."""
    ref_k_e = ref_cav.get('k_e', to_numpy=False)

    if global_compensation:
        logging.warning("Limits for the electric field were manually set to "
                        + "a very low value in order to be consistent with "
                        + "the 'global' or 'global downstream' compensation "
                        + "strategy that you asked for.")
        lower = ref_k_e
        upper = ref_k_e * 1.000001
        return (lower, upper)

    if preset == 'MYRRHA':
        if ref_elts is None:
            logging.error("The reference ListOfElements is required for MYRRHA"
                          " preset.")
            return (None, None)

        # Minimum: reference - 50%
        lower = ref_k_e * 0.5

        # Maximum: maximum of section + 30%
        this_section = ref_cav.idx['section']
        k_e_this_section = [cav.get('k_e', to_numpy=False)
                            for cav in ref_elts
                            if cav.idx['section'] == this_section]
        upper = np.nanmax(k_e_this_section) * 1.3

        return (lower, upper)

    if preset == 'JAEA':
        # Minimum: reference - 50%
        lower = ref_k_e * 0.5

        # Maximum: reference + 20%
        upper = ref_k_e * 1.2

        return (lower, upper)

    logging.error("k_e has no limits implemented for the preset "
                  f"{preset}. Check optimisation.parameters.factories module.")
    return (None, None)


def _limits_phi_0(**kwargs) -> tuple[float | None]:
    """Limits for the relative or absolute cavity phase."""
    return (0, 4 * np.pi)


def _limits_phi_s(**kwargs) -> tuple[float | None]:
    """
    Limits for the synchrous phase.

    Used when you want to set constraints on the synchronous phase but the
    optimisation algorithm does not support it. In this case, phi_s is
    considered as a variable (needs an additional optimisation loop to find the
    phi_0 corresponding to the phi_s asked by optimisation algorithm).

    """
    return _constraints_phi_s(**kwargs)


LIMITS_CALCULATORS = {
    'k_e': _limits_k_e,
    'phi_0_rel': _limits_phi_0,
    'phi_0_abs': _limits_phi_0,
    'phi_s': _limits_phi_s,
}


# =============================================================================
# Presets for variable initial values
# =============================================================================
INITIAL_VALUE_CALCULATORS = {
    'k_e': lambda cav: cav.get('k_e', to_numpy=False),
    'phi_0_rel': lambda cav: 0.,
    'phi_0_abs': lambda cav: 0.,
    'phi_s': lambda cav: cav.get('phi_s', to_numpy=False),
}


# =============================================================================
# Presets, helpers for constraints
# =============================================================================
def _constraints_phi_s(preset: str | None = None,
                       ref_cav: FieldMap | None = None,
                       **kwargs) -> tuple[float | None]:
    """Set the constraints on the synchronous phase."""
    ref_phi_s = ref_cav.get('phi_s', to_numpy=False)
    if preset == 'MYRRHA':
        # Minimum: -90deg
        lower = -np.pi / 2.

        # Maximum: 0deg or reference + 40%           (reminder: phi_s < 0)
        upper = min(0., ref_phi_s * (1. - 0.4))

        return (lower, upper)

    if preset == 'JAEA':
        # Minimum: -90deg
        lower = -np.pi / 2.

        # Maximum: 0deg or reference + 50%           (reminder: phi_s < 0)
        upper = min(0., ref_phi_s * (1. - 0.5))

        return (lower, upper)

    logging.error("phi_s has no constraints implemented for the preset "
                  f"{preset}. Check optimisation.parameters.factories module.")
    return (None, None)


CONSTRAINTS_CALCULATORS = {
    'phi_s': _constraints_phi_s
}


def _compute_constraints(constraints: list[Constraint],
                         simulation_output: SimulationOutput) -> np.ndarray:
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
# Helper for objectives
# =============================================================================
def _compute_residuals(objectives: list[Objective],
                       simulation_output: SimulationOutput) -> np.ndarray:
    """Compute residuals on given `Objectives` for given `SimulationOutput`."""
    residuals = [objective.evaluate(simulation_output)
                 for objective in objectives]
    return np.array(residuals)
