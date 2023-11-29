#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds a :class:`FieldMap`.

.. todo::
    Handle the different kind of field_maps...

.. todo::
    Handle the SET_SYNCH_PHASE command

.. todo::
    Hande phi_s fitting with :class:`beam_calculation.tracewin.Tracewin`

.. todo::
    when subclassing field_maps, do not forget to update the transfer matrix
    selector in:
    - :class:`.Envelope3D`
    - :class:`.SingleElementEnvelope3DParameters`
    - :class:`.SetOfCavitySettings`
    - the ``run_with_this`` methods

"""
import logging
from pathlib import Path
from typing import Any

import numpy as np

from scipy.optimize import minimize_scalar

from core.elements.element import Element
import config_manager as con

from core.electric_field import (phi_0_rel_corresponding_to, RfField,
                                 phi_0_abs_corresponding_to)

from util.helper import diff_angle

from failures.set_of_cavity_settings import SingleCavitySettings


IMPLEMENTED_STATUS = (
    # Cavity settings not changed from .dat
    "nominal",
    # Cavity ABSOLUTE phase changed; relative phase unchanged
    "rephased (in progress)",
    "rephased (ok)",
    # Cavity norm is 0
    "failed",
    # Trying to fit
    "compensate (in progress)",
    # Compensating, proper setting found
    "compensate (ok)",
    # Compensating, proper setting not found
    "compensate (not ok)",
)  #:


class FieldMap(Element):
    """A generic ``FIELD_MAP``."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 default_field_map_folder: Path,
                 elt_name: str | None = None,
                 **kwargs) -> None:
        """Set most of attributes defined in ``TraceWin``."""
        super().__init__(line, dat_idx, elt_name)
        n_attributes = len(line) - 1
        assert n_attributes == 10

        self.geometry = int(line[1])
        self.length_m = 1e-3 * float(line[2])
        self.aperture_flag = int(line[8])               # K_a

        # handled by TW
        # if self.aperture_flag > 0:
        #     logging.warning("Space charge compensation maps not handled.")
        # FIXME according to doc, may also be float

        self.field_map_folder = default_field_map_folder
        self.field_map_file_name: Path | list[Path]
        self._prepare_field_map(line)
        self.update_status('nominal')

    @property
    def is_accelerating(self) -> bool:
        """Tell if the cavity is working."""
        if self.elt_info['status'] == 'failed':
            return False
        # if self.acc_field.k_e < 1e-8:
        #     return False
        return True

    @property
    def can_be_retuned(self) -> bool:
        """Tell if we can modify the element's tuning."""
        return True

    def _prepare_field_map(self, line: list[str]) -> None:
        """Set field map related parameters.

        Field map file(s) are not loaded at initialization, but rather once all
        :class:`.Command` and in particular :class:`.FieldMapPath` have been
        dealt with.

        Parameters
        ----------
        line : list[str]
            Full line corresponding to current field map.

        """
        phi_0 = np.deg2rad(float(line[3]))
        self.field_map_file_name = Path(line[9])
        absolute_phase_flag = bool(int(line[10]))
        self.acc_field = RfField(k_e=float(line[6]),
                                 absolute_phase_flag=absolute_phase_flag,
                                 phi_0=phi_0)

    def update_status(self, new_status: str) -> None:
        """
        Change the status of a cavity.

        We also ensure that the value new_status is correct. If the new value
        is 'failed', we also set the norm of the electric field to 0.

        """
        assert new_status in IMPLEMENTED_STATUS

        self.elt_info['status'] = new_status
        if new_status == 'failed':
            self.acc_field.k_e = 0.
            for beam_calc_param in self.beam_calc_param.values():
                beam_calc_param.re_set_for_broken_cavity()

    def set_full_path(self, extensions: dict[str, list[str]]) -> None:
        """
        Set absolute paths with extensions of electromagnetic files.

        Parameters
        ----------
        extensions : dict[str, list[str]]
            Keys are nature of the field, values are a list of extensions
            corresponding to it without a period.

        See also
        --------
        :func:`tracewin_utils.electromagnetic_fields.file_map_extensions`

        """
        self.field_map_file_name = [
            Path(self.field_map_folder,
                 self.field_map_file_name).with_suffix('.' + ext)
            for extension in extensions.values()
            for ext in extension
        ]

    def rf_param(self,
                 solver_id: str,
                 phi_bunch_abs: float,
                 w_kin_in: float,
                 cavity_settings: SingleCavitySettings | None = None) -> dict:
        """
        Set the properties of the rf field.

        Parameters
        ----------
        solver_id : str
            Identificator of the :class:`.BeamCalculator`.
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
        if status in ('none', 'failed'):
            return {}

        generic_rf_param = {
            'omega0_rf': self.get('omega0_rf'),
            'e_spat': self.acc_field.e_spat,
            'section_idx': self.idx['section'],
            'n_cell': self.get('n_cell'),
            'bunch_to_rf': self.get('bunch_to_rf'),
        }

        if status in ('nominal', 'rephased (ok)', 'compensate (ok)',
                      'compensate (not ok)'):
            norm_and_phases, abs_to_rel = _get_from(self.acc_field)

        elif status in ('rephased (in progress)'):
            norm_and_phases, abs_to_rel = _get_from(self.acc_field,
                                                    force_rephasing=True)

        elif status in ('compensate (in progress)'):
            assert isinstance(cavity_settings, SingleCavitySettings)
            norm_and_phases, abs_to_rel = _try_this(solver_id,
                                                    cavity_settings,
                                                    w_kin_in,
                                                    self,
                                                    generic_rf_param)
        else:
            logging.error(f'{self} {status = } is not allowed.')
            return {}

        phi_rf_abs = phi_bunch_abs * generic_rf_param['bunch_to_rf']

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
    def phi_0_rel_matching_this(self,
                                solver_id: str,
                                phi_s: float,
                                w_kin_in: float,
                                **rf_parameters: dict
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
            results = beam_calc_param.transf_mat_function_wrapper(
                w_kin_in,
                **rf_parameters)
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
