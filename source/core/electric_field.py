#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:43:39 2021.

@author: placais

This module holds RfField, which is a simple electric field.

TODO : untouched for many time... Should modernize this module, maybe move it
to beam_calculation/?

"""
import logging
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
    All phases are stored as radian.

    Attributes
    ----------
    e_spat : Callable[float, float]
        Spatial component of the electric field. Needs to be multiplied by the
        cos(omega t) to have the full electric field. Initialized to null
        function.
    k_e : float
        Norm of the electric field.
    phi_0 : dict[str, None | float | bool]
        Holds the electric field phase. The keys are:
            phi_0_rel : relative phi_0 in rad
            phi_0_abs : absolute phi_0 in rad
            nominal_rel : relative phi_0 in rad in the nominal (ref) linac
            abs_phase_flag : if the relative or absolute phi_0 must be used
            phi_0_abs_but_reference_phase_is_different : used when the
            `ListOfElements` under study does not start at the beginning of the
            linac and we use TraceWin.
            new_reference_phase : phase at the entrance of this
            `ListOfElements`.
    v_cav_mv : float
        Cavity accelerating field in MV.
    phi_s : float
        Cavity synchronous phase in rad.
    omega0_rf : float | None
        RF pulsation of the cavity in rad/s.
    n_cell : int | None
        Number of cells in the cavity.
    n_z : int | None
        Number of points in the file that gives `e_spat`, the spatial component
        of the electric field.

    """

    def __init__(self, k_e: float = np.NaN, absolute_phase_flag: bool = False,
                 phi_0: float = None) -> None:
        self.e_spat = lambda x: 0.
        self.k_e = k_e

        self.phi_0 = {'phi_0_rel': None,
                      'phi_0_abs': None,
                      'nominal_rel': None,
                      'abs_phase_flag': absolute_phase_flag,
                      'phi_0_abs_but_reference_phase_is_different': None,
                      'new_reference_phase': None,
                      }

        if absolute_phase_flag:
            self.phi_0['phi_0_abs'] = phi_0
        else:
            self.phi_0['phi_0_rel'] = phi_0
            self.phi_0['nominal_rel'] = phi_0

        self.v_cav_mv = np.NaN
        self.phi_s = np.NaN

        # Initialized later as it depends on the Section the cavity is in
        self.omega0_rf, self.n_cell = None, None

        # Depends on beam_computer, but also on n_cell
        self.n_z = None

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_deg: bool = False, **kwargs: bool | str
            ) -> list | np.ndarray | float | None:
        """
        Shorthand to get attributes from this class.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_deg : bool, optional
            To apply np.rad2deg function over every `key` containing the string
            'phi'. The default is False.
        **kwargs : bool | str
            Other arguments passed to recursive getter.

        Returns
        -------
        out : list | np.ndarray | float | None
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)

            if val[key] is not None and to_deg and 'phi' in key:
                val[key] = np.rad2deg(val[key])

        out = [val[key] for key in keys]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def set_pulsation_ncell(self, f_mhz: float, n_cell: int) -> None:
        """Initialize the frequency and the number of cells."""
        self.omega0_rf = 2e6 * np.pi * f_mhz
        self.n_cell = n_cell

    # FIXME : still used? nominal_rel still used?
    def rephase_cavity(self, phi_rf_abs: float) -> None:
        """
        Rephase cavity to ensure that `phi_0_rel` is the same as nominal case.

        In other words, we want the particle to enter with the same relative
        phase as in the nominal linac.

        """
        assert self.phi_0['nominal_rel'] is not None
        self.phi_0['phi_0_rel'] = self.phi_0['nominal_rel']
        self.phi_0['phi_0_abs'] = phi_0_abs_corresponding_to(
            self.phi_0['phi_0_rel'],
            phi_rf_abs
        )

    def update_phi_0_abs_to_adapt_to_new_ref_phase(
            self,
            old_phi_in: float,
            new_phi_in: float,
            phases_are_bunch: bool = True,
    ) -> float:
        """Calculate the new `phi_0_abs`, with a new reference phase."""
        if not self.phi_0['abs_phase_flag']:
            logging.error("For this cavity we use the relative phi_0. Why do "
                          "you want to change the absolute phase of the cav? "
                          "Returning the relative phi_0, which should allow "
                          "LightWin to continue its execution...")
            return self.phi_0['phi_0_rel']

        delta_phi_rf = new_phi_in - old_phi_in

        if phases_are_bunch:
            delta_phi_rf *= self.n_cell
            new_phi_in *= self.n_cell

        new_phi_0_abs = phi_0_abs_with_new_phase_reference(
            self.phi_0['phi_0_abs'],
            delta_phi_rf
        )

        self.phi_0['new_reference_phase'] = new_phi_in
        self.phi_0['phi_0_abs_but_reference_phase_is_different'] = \
            new_phi_0_abs
        return new_phi_0_abs


# =============================================================================
# Helper functions dedicated to electric fields
# =============================================================================
def phi_0_rel_corresponding_to(phi_0_abs: float, phi_rf_abs: float) -> float:
    """Calculate a cavity relative entrance phase from the absolute."""
    return np.mod(phi_0_abs + phi_rf_abs, 2. * np.pi)


def phi_0_abs_corresponding_to(phi_0_rel: float, phi_rf_abs: float) -> float:
    """Calculate a cavity absolute entrance phase from the relative."""
    return np.mod(phi_0_rel - phi_rf_abs, 2. * np.pi)


def phi_0_abs_with_new_phase_reference(phi_0_abs: float, delta_phi_rf: float,
                                       ) -> float:
    """
    Calculate a new absolute entrance phase, when the ref phase changed.

    Parameters
    ----------
    phi_0_abs : float
        Absolute entry phase of the cavity.
    delta_phi_rf : float
        Change in rf phase (not bunch!). Usually defined as `new_phi - old_phi`
        where `old_phi` is the entry phase for which `phi_0_abs` is valid.

    Returns
    -------
    phi_0_abs : float
        A new absolute cavity entry phase. The reference phase is now `new_phi`
        so that calculations will be valid if a different `ListOfElements` is
        calculated.

    """
    return np.mod(delta_phi_rf + phi_0_abs, 2. * np.pi)
