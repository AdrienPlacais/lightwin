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
from pathlib import Path

import numpy as np

from core.electric_field import RfField
from core.elements.element import Element
from core.elements.field_maps.cavity_settings import CavitySettings


# warning: doublon with cavity_settings.ALLOWED_STATUS
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
                 cavity_settings: CavitySettings,
                 name: str | None = None,
                 **kwargs) -> None:
        """Set most of attributes defined in ``TraceWin``."""
        super().__init__(line, dat_idx, name)
        n_attributes = len(line) - 1
        assert n_attributes == 10

        self.geometry = int(line[1])
        self.length_m = 1e-3 * float(line[2])
        self.aperture_flag = int(line[8])               # K_a
        self.cavity_settings = cavity_settings

        # handled by TW
        # if self.aperture_flag > 0:
        #     logging.warning("Space charge compensation maps not handled.")
        # FIXME according to doc, may also be float

        self.field_map_folder = default_field_map_folder
        self.field_map_file_name: Path | list[Path]

        # wont be necessary anymore
        self._prepare_field_map(line)
        self.rf_field: RfField
        self.update_status('nominal')

        # will have to set this at instantiation I guess

    # maybe I should get this from CavitySettings instead
    @property
    def is_accelerating(self) -> bool:
        """Tell if the cavity is working."""
        if self.elt_info['status'] == 'failed':
            return False
        # if self.rf_field.k_e < 1e-8:
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
        self.rf_field = RfField(k_e=float(line[6]),
                                absolute_phase_flag=absolute_phase_flag,
                                phi_0=phi_0)

    def update_status(self, new_status: str) -> None:
        """
        Change the status of a cavity.

        We also ensure that the value new_status is correct. If the new value
        is 'failed', we also set the norm of the electric field to 0.

        """
        assert new_status in IMPLEMENTED_STATUS

        self.cavity_settings.status = new_status
        self.elt_info['status'] = new_status
        if new_status == 'failed':
            self.rf_field.k_e = 0.
            for solver_id, beam_calc_param in self.beam_calc_param.items():
                beam_calc_param.re_set_for_broken_cavity()
                self.cavity_settings.set_beam_calculator(
                    solver_id,
                    beam_calc_param.transf_mat_function_wrapper)

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
