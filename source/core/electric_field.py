#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:43:39 2021.

@author: placais
"""
import cmath
import numpy as np

from util.helper import recursive_items, recursive_getter


def compute_param_cav(integrated_field, status):
    """Compute synchronous phase and accelerating field."""
    if status == 'failed':
        polar_itg = np.array([np.NaN, np.NaN])
    else:
        polar_itg = cmath.polar(integrated_field)
    cav_params = {'v_cav_mv': polar_itg[0],
                  'phi_s': polar_itg[1]}
    return cav_params


class RfField():
    """
    Cos-like RF field.

    Warning, all phases are defined as:
        phi = omega_0_rf * t
    While in the rest of the code and in particular in Particle, it is defined
    as:
        phi = omega_0_bunch * t
    """

    def __init__(self, k_e: float = np.NaN, absolute_phase_flag: bool = False,
                 phi_0: float = None) -> None:
        # By default, electric field spatial function is null.
        self.e_spat = lambda x: 0.
        self.k_e = k_e

        self.phi_0 = {'phi_0_rel': None,
                      'phi_0_abs': None,
                      'nominal_rel': None,
                      'abs_phase_flag': absolute_phase_flag}

        if absolute_phase_flag:
            self.phi_0['phi_0_abs'] = phi_0
        else:
            self.phi_0['phi_0_rel'] = phi_0
            self.phi_0['nominal_rel'] = phi_0

        # self.cav_params = {'v_cav_mv': np.NaN, 'phi_s': np.NaN}
        self.v_cav_mv = np.NaN
        self.phi_s = np.NaN

        # Initialized later as it depends on the Section the cavity is in
        self.omega0_rf, self.n_cell = None, None

        # Depends on beam_computer, but also on n_cell
        self.n_z = None

    def has(self, key):
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys, to_deg=False, **kwargs):
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)

            if val[key] is not None and to_deg and 'phi' in key:
                val[key] = np.rad2deg(val[key])

        # Convert to list
        out = [val[key] for key in keys]

        if len(out) == 1:
            return out[0]
        # implicit else:
        return tuple(out)

    def set_pulsation_ncell(self, f_mhz, n_cell):
        """Initialize the frequency and the number of cells."""
        self.omega0_rf = 2e6 * np.pi * f_mhz
        self.n_cell = n_cell

    def rephase_cavity(self, phi_rf_abs):
        """
        Rephase the cavity.

        In other words, we want the particle to enter with the same relative
        phase as in the nominal linac.
        """
        assert self.phi_0['nominal_rel'] is not None
        self.phi_0['phi_0_rel'] = self.phi_0['nominal_rel']
        self.phi_0['phi_0_abs'] = np.mod(
            self.phi_0['nominal_rel'] - phi_rf_abs, 2. * np.pi)


# =============================================================================
# Helper functions dedicated to electric fields
# =============================================================================
def phi_0_rel_corresponding_to(phi_0_abs: float, phi_rf_abs: float) -> float:
    """Calculate a cavity relative entrance phase from the absolute."""
    return np.mod(phi_0_abs + phi_rf_abs, 2. * np.pi)


def phi_0_abs_corresponding_to(phi_0_rel: float, phi_rf_abs: float) -> float:
    """Calculate a cavity absolute entrance phase from the relative."""
    return np.mod(phi_0_rel - phi_rf_abs, 2. * np.pi)
