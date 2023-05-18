#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:43:55 2023

@author: placais
"""
import logging
import numpy as np

from core.accelerator import Accelerator
from core.elements import FieldMap


def initial_value(key: str, ref_cav: FieldMap) -> float | None:
    """Return initial guess for desired key."""
    if key not in D_INITIAL:
        logging.error(f"Initial value for variable {key} not implemented.")
        return None
    return D_INITIAL[key](ref_cav)


def limits(key: str, *args: tuple[str, FieldMap, FieldMap, Accelerator]
           ) -> tuple[float | None]:
    """Return optimisation limits for desired key."""
    if key not in D_LIM:
        logging.error(f"Limits for variable {key} not implemented.")
        return (None, None)
    return D_LIM[key](*args)


def constraints(key: str, *args: tuple[str, FieldMap, FieldMap, Accelerator]
                ) -> tuple[float | None]:
    """Return optimisation constraints for desired key."""
    if key not in D_CONST:
        logging.error(f"Constraint for variable {key} not implemented.")
        return (None, None)
    return D_CONST[key](*args)


def _limits_k_e(preset: str, cav: FieldMap, ref_cav: FieldMap, ref_linac:
                Accelerator) -> tuple[float | None]:
    """Limits for electric field."""
    ref_k_e = ref_cav.get('k_e', to_numpy=False)
    if preset == 'MYRRHA':
        if ref_linac is None:
            logging.error("The reference linac is required for MYRRHA preset.")
            return (None, None)

        # Minimum: reference - 50%
        lower = ref_k_e * 0.5

        # Maximum: maximum of section + 30%
        this_section = cav.idx['section']
        cavs_this_section = ref_linac.l_cav
        k_e_this_section = [cav.get('k_e', to_numpy=False)
                            for cav in cavs_this_section
                            if cav.idx['section'] == this_section]
        upper = np.max(k_e_this_section) * 1.3

        logging.warning("Manually modified the k_e limits for global comp.")
        lower = ref_k_e
        upper = ref_k_e * 1.000001

        return (lower, upper)

    if preset == 'JAEA':
        # Minimum: reference - 50%
        lower = ref_k_e * 0.5

        # Maximum: reference + 20%
        upper = ref_k_e * 1.2

        return (lower, upper)

    logging.error(f"Preset {preset} not implemented!")
    return (None, None)


def _limits_phi_0(preset: str, cav: FieldMap, ref_cav: FieldMap,
                  ref_linac: Accelerator) -> tuple[float | None]:
    """Limits for the relative or absolute cavity phase."""
    return (0, 4 * np.pi)


def _limits_phi_s(preset: str, cav: FieldMap, ref_cav: FieldMap,
                  ref_linac: Accelerator) -> tuple[float | None]:
    """Limits for the synchrous phase; also used to set phi_s constraints."""
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

    logging.error(f"Preset {preset} not implemented!")
    return (None, None)


D_INITIAL = {
    'k_e': lambda cav: cav.get('k_e', to_numpy=False),
    'phi_0_rel': lambda cav: 0.,
    'phi_0_abs': lambda cav: 0.,
    'phi_s': lambda cav: cav.get('phi_s', to_numpy=False),
}

D_LIM = {
    'k_e': _limits_k_e,
    'phi_0_rel': _limits_phi_0,
    'phi_0_abs': _limits_phi_0,
    'phi_s': _limits_phi_s,
}

D_CONST = {
    'phi_s': _limits_phi_s
}
