#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds a :class:`FieldMap`.

.. todo::
    Handle the different kind of field_maps...

.. todo::
    Handle the SET_SYNCH_PHASE command

.. todo::
    Hande phi_s fitting with :class:`beam_calculation.tracewin.Tracewin`

"""
import logging
from typing import Any
import numpy as np

from scipy.optimize import minimize_scalar

from core.elements.element import Element
import config_manager as con

from core.electric_field import (phi_0_rel_corresponding_to, RfField,
                                 phi_0_abs_corresponding_to)

from util.helper import diff_angle

from failures.set_of_cavity_settings import SingleCavitySettings


class FieldMap(Element):
    """A generic ``FIELD_MAP``."""

    def __init__(self, line: list[str]) -> None:
        n_attributes = len(line) - 1
        assert n_attributes in [9, 10]

        super().__init__(line)
        self.geometry = int(line[1])
        self.length_m = 1e-3 * float(line[2])
        self.aperture_flag = int(line[8])               # K_a
        # FIXME according to doc, may also be float

        self.field_map_file_name = str(line[9])         # FileName
        self.field_map_folder: str

        if len(line) == 10:
            line.append('1')
        absolute_phase_flag = int(line[10])

        self.acc_field = RfField(k_e=float(line[6]),
                                 absolute_phase_flag=bool(absolute_phase_flag),
                                 phi_0=np.deg2rad(float(line[3])))
        self.update_status('nominal')

    def rf_param(self, solver_id: str, phi_bunch_abs: float, w_kin_in: float,
                 cavity_settings: SingleCavitySettings | None = None) -> dict:
        """
        Set the properties of the rf field; specific to FieldMap.

        Parameters
        ----------
        solver_id : str
        Identificator of the :class:`BeamCalculator`.
        phi_bunch_abs : float
            Absolute phase of the particle (bunch frequency).
        w_kin_in : float
            Kinetic energy at the element entrance in MeV.
        cavity_settings : SingleCavitySettings | None, optional
            Cavity settings. Should be None in a non-accelerating element such
            as a Drift or a broken FieldMap, and in accelerating elements
            outside the fit process. The default is None.

        Returns
        -------
        rf_parameters : dict
            Holds parameters that Envelope1d needs to solve the motion in the
            FieldMap. If this is a non-accelerating element, return {}.

        """
        status = self.elt_info['status']
        if status in ['none', 'failed']:
            return {}

        generic_rf_param = {
            'omega0_rf': self.get('omega0_rf'),
            'e_spat': self.acc_field.e_spat,
            'section_idx': self.idx['section'],
            'n_cell': self.get('n_cell')
        }

        if status in ('nominal', 'rephased (ok)', 'compensate (ok)',
                      'compensate (not ok)'):
            norm_and_phases, abs_to_rel = _get_from(self.acc_field)

        elif status in ('rephased (in progress)'):
            norm_and_phases, abs_to_rel = _get_from(self.acc_field,
                                                    force_rephasing=True)

        elif status in ('compensate (in progress)'):
            assert isinstance(cavity_settings, SingleCavitySettings)
            norm_and_phases, abs_to_rel = \
                _try_this(solver_id, cavity_settings, w_kin_in, self,
                          generic_rf_param)
        else:
            logging.error(f'{self} {status = } is not allowed.')
            return {}

        phi_rf_abs = phi_bunch_abs * generic_rf_param['n_cell']

        if abs_to_rel:
            norm_and_phases['phi_0_rel'] = phi_0_rel_corresponding_to(
                norm_and_phases['phi_0_abs'], phi_rf_abs)
        else:
            norm_and_phases['phi_0_abs'] = phi_0_abs_corresponding_to(
                norm_and_phases['phi_0_rel'], phi_rf_abs)

        # '|' merges the two dictionaries
        rf_parameters = generic_rf_param | norm_and_phases
        return rf_parameters

    # FIXME to refactor, in particular send proper beam_parameters directly.
    def phi_0_rel_matching_this(self, solver_id: str, phi_s: float,
                                w_kin_in: float, **rf_parameters: dict
                                ) -> float:
        """
        Sweeps phi_0_rel until the cavity synch phase matches phi_s.

        Parameters
        ----------
        solver_id : str
            BeamCalculator identificator.
        phi_s : float
            Synchronous phase to be matched.
        w_kin_in : float
            Kinetic energy at the cavity entrance in MeV.
        rf_parameters : dict
            Holds all rf electric field parameters.

        Return
        ------
        phi_0_rel : float
            The relative cavity entrance phase that leads to a synchronous
            phase of phi_s_objective.
        """
        bounds = (0, 2. * np.pi)
        beam_calc_param = self.beam_calc_param[solver_id]

        def _wrapper_synch(phi_0_rel: float) -> float:
            rf_parameters['phi_0_rel'] = phi_0_rel
            rf_parameters['phi_0_abs'] = None
            # FIXME should not have dependencies is_accelerating, status
            args = (w_kin_in, self.is_accelerating(), self.get('status'))
            results = beam_calc_param.transf_mat_function_wrapper(
                *args, **rf_parameters)
            diff = diff_angle(phi_s, results['cav_params']['phi_s'])
            return diff**2

        res = minimize_scalar(_wrapper_synch, bounds=bounds)
        if not res.success:
            logging.error('Synch phase not found')
        phi_0_rel = res.x
        return phi_0_rel


def _get_from(rf_field: RfField, force_rephasing: bool = False
              ) -> tuple[dict, bool]:
    """Get norms and phases from RfField object."""
    norm_and_phases = {
        'k_e': rf_field.get('k_e'),
        'phi_0_rel': None,
        'phi_0_abs': rf_field.get('phi_0_abs')
    }
    abs_to_rel = True

    # If we are calculating the transfer matrices of the nominal linac and the
    # initial phases are defined in the .dat as relative phases, phi_0_abs is
    # not defined yet
    if norm_and_phases['phi_0_abs'] is None or force_rephasing:
        norm_and_phases['phi_0_rel'] = rf_field.get('phi_0_rel')
        abs_to_rel = False
    return norm_and_phases, abs_to_rel


# FIXME to refactor
def _try_this(solver_id: str, cavity_settings: SingleCavitySettings,
              w_kin: float, cav: FieldMap, generic_rf_param: dict[str, Any],
              ) -> tuple[dict, bool]:
    """Extract parameters from cavity_parameters."""
    if cavity_settings.phi_s is not None:
        generic_rf_param['k_e'] = cavity_settings.k_e
        phi_0_rel = cav.phi_0_rel_matching_this(
            solver_id, cavity_settings.phi_s, w_kin_in=w_kin,
            **generic_rf_param)

        norm_and_phases = {
            'k_e': cavity_settings.k_e,
            'phi_0_rel': phi_0_rel,
        }
        abs_to_rel = False
        return norm_and_phases, abs_to_rel

    norm_and_phases = {
        'k_e': cavity_settings.k_e,
        'phi_0_abs': cavity_settings.phi_0_abs,
        'phi_0_rel': cavity_settings.phi_0_rel,
    }
    abs_to_rel = con.FLAG_PHI_ABS
    return norm_and_phases, abs_to_rel
