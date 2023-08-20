#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:44:41 2023.

@author: placais
"""
import logging

import numpy as np

from optimisation.parameters.variable import Variable
from optimisation.parameters.constraint import Constraint

from core.list_of_elements import ListOfElements, equiv_elt
from core.elements import FieldMap


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
                       ) -> list[Constraint]:
    """Create the necessary `Constraint` objects."""
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
    return constraints


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
# Presets for constraints
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
