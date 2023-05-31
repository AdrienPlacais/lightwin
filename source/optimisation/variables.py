#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 16:43:55 2023.

@author: placais
"""
import logging
from dataclasses import dataclass
import numpy as np

from core.accelerator import Accelerator
from core.elements import FieldMap
from util.dicts_output import d_markdown


@dataclass
class Variable:
    """A single variable."""
    name: str
    cavity_name: str
    x_0: float
    limits: tuple

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        self.x_0_fmt, self.limits_fmt = self.x_0, self.limits
        if 'phi' in self.name:
            self.x_0_fmt = np.rad2deg(self.x_0)
            self.limits_fmt = np.rad2deg(self.limits)

    def __str__(self):
        out = f"{d_markdown[self.name]:20} {self.cavity_name:5} "
        out += f"x_0={self.x_0_fmt:>8.3f}   "
        out += f"limits={self.limits_fmt[0]:>8.3f} {self.limits_fmt[1]:>8.3f}"
        return out


@dataclass
class Constraint:
    """A single constraint."""
    name: str
    cavity_name: str
    limits: tuple

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        self.limits_fmt = self.limits
        if 'phi' in self.name:
            self.limits_fmt = np.rad2deg(self.limits)

    def __str__(self):
        out = f"{d_markdown[self.name]:20} {self.cavity_name:15}      "
        out += f"limits={self.limits_fmt[0]:>8.3f} {self.limits_fmt[1]:>8.3f}"
        return out


class VariablesAndConstraints:
    """Holds variables, constraints, bounds of the optimisation problem."""

    def __init__(self, accelerator_name: str, ref_acc: Accelerator,
                 comp_cav: list[FieldMap], variable_names: list[str],
                 constraint_names: list[str]) -> None:
        """Set the design space."""
        self.accelerator_name = accelerator_name
        self.ref_acc = ref_acc
        self.comp_cav = comp_cav
        self.variable_names = variable_names
        self.constraint_names = constraint_names

        self.variables = [Variable(name=var, cavity_name=str(cav),
                                   x_0=self._set_initial_value(var, cav),
                                   limits=self._set_limits(var, cav))
                          for var in self.variable_names
                          for cav in self.comp_cav]
        self.constraints = [Constraint(name=con, cavity_name=str(cav),
                                       limits=self._set_constraints(con, cav))
                            for con in self.constraint_names
                            for cav in self.comp_cav]

    def __str__(self) -> str:
        out = ["=" * 80]
        out += ["Variables:"] + [str(var) for var in self.variables]
        out += ["-" * 80]
        out += ["Constraints (not used with least squares):"]
        out += [str(con) for con in self.constraints]
        out += ["=" * 80]
        return "\n".join(out)

    # TODO legacy
    def to_least_squares_format(self) -> tuple[np.ndarray, np.ndarray,
                                               np.ndarray, list[str]]:
        """Return design space as expected by scipy.least_squares."""
        x_0 = np.array([var.x_0 for var in self.variables])
        x_lim = np.array([var.limits for var in self.variables])
        g_lim = np.array([con.limits for con in self.constraints])
        l_x_str = str(self)
        return x_0, x_lim, g_lim, l_x_str

    def _set_initial_value(self, key: str, cav: FieldMap) -> float | None:
        """Return initial guess for desired key."""
        if key not in INITIAL:
            logging.error(f"Initial value for variable {key} not implemented.")
            return None
        ref_cav = self.ref_acc.equiv_elt(cav)
        return INITIAL[key](ref_cav)

    def _set_limits(self, key: str, cav: FieldMap) -> tuple[float | None]:
        """Return optimisation limits for desired key."""
        if key not in LIM:
            logging.error(f"Limits for variable {key} not implemented.")
            return (None, None)
        ref_cav = self.ref_acc.equiv_elt(cav)
        args = (self.accelerator_name, cav, ref_cav, self.ref_acc)
        return LIM[key](*args)

    def _set_constraints(self, key: str, cav: FieldMap) -> tuple[float | None]:
        """Return optimisation constraints for desired key."""
        if key not in CONST:
            logging.error(f"Constraint for variable {key} not implemented.")
            return (None, None)
        ref_cav = self.ref_acc.equiv_elt(cav)
        args = (self.accelerator_name, cav, ref_cav, self.ref_acc)
        return CONST[key](*args)


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


INITIAL = {
    'k_e': lambda cav: cav.get('k_e', to_numpy=False),
    'phi_0_rel': lambda cav: 0.,
    'phi_0_abs': lambda cav: 0.,
    'phi_s': lambda cav: cav.get('phi_s', to_numpy=False),
}

LIM = {
    'k_e': _limits_k_e,
    'phi_0_rel': _limits_phi_0,
    'phi_0_abs': _limits_phi_0,
    'phi_s': _limits_phi_s,
}

CONST = {
    'phi_s': _limits_phi_s
}
