#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold a ``FIELD_MAP``.

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
from typing import Any

import numpy as np

from core.electric_field import NewRfField
from core.elements.element import Element
from core.elements.field_maps.cavity_settings import CavitySettings
from util.helper import recursive_getter

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

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        default_field_map_folder: Path,
        cavity_settings: CavitySettings,
        name: str | None = None,
        **kwargs,
    ) -> None:
        """Set most of attributes defined in ``TraceWin``."""
        super().__init__(line, dat_idx, name)
        n_attributes = len(line) - 1
        assert n_attributes == 10

        self.geometry = int(line[1])
        self.length_m = 1e-3 * float(line[2])
        self.aperture_flag = int(line[8])  # K_a
        self.cavity_settings = cavity_settings

        self.field_map_folder = default_field_map_folder
        self.field_map_file_name = Path(line[9])

        self.new_rf_field: NewRfField

    @property
    def status(self) -> str:
        """Give the status from the :class:`.CavitySettings`."""
        return self.cavity_settings.status

    @property
    def is_accelerating(self) -> bool:
        """Tell if the cavity is working."""
        if self.status == "failed":
            return False
        return True

    @property
    def can_be_retuned(self) -> bool:
        """Tell if we can modify the element's tuning."""
        return True

    def update_status(self, new_status: str) -> None:
        """Change the status of the cavity.

        We use
        :meth:`.ElementBeamCalculatorParameters.re_set_for_broken_cavity`
        method.
        If ``k_e``, ``phi_s``, ``v_cav_mv`` are altered, this is performed in
        :meth:`.CavitySettings.status` ``setter``.

        """
        assert new_status in IMPLEMENTED_STATUS

        self.cavity_settings.status = new_status
        if new_status != "failed":
            return

        for solver_id, beam_calc_param in self.beam_calc_param.items():
            new_transf_mat_func = beam_calc_param.re_set_for_broken_cavity()
            self.cavity_settings.set_cavity_parameters_methods(
                solver_id,
                new_transf_mat_func,
            )
        return

    def set_full_path(self, extensions: dict[str, list[str]]) -> None:
        """Set absolute paths with extensions of electromagnetic files.

        Parameters
        ----------
        extensions : dict[str, list[str]]
            Keys are nature of the field, values are a list of extensions
            corresponding to it without a period.

        See Also
        --------
        :func:`tracewin_utils.electromagnetic_fields.file_map_extensions`

        """
        self.field_map_file_name = [
            Path(self.field_map_folder, self.field_map_file_name).with_suffix(
                "." + ext
            )
            for extension in extensions.values()
            for ext in extension
        ]

    def keep_cavity_settings(self, cavity_settings: CavitySettings) -> None:
        """Keep the cavity settings that were found."""
        assert cavity_settings is not None
        self.cavity_settings = cavity_settings

    def get(
        self,
        *keys: str,
        to_numpy: bool = True,
        none_to_nan: bool = False,
        **kwargs: bool | str | None,
    ) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        **kwargs : bool | str | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if key == "name":
                val[key] = self.name
                continue

            if self.cavity_settings.has(key):
                val[key] = self.cavity_settings.get(key)
                continue

            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        out = [
            (
                np.array(val[key])
                if to_numpy and not isinstance(val[key], str)
                else val[key]
            )
            for key in keys
        ]
        if none_to_nan:
            out = [x if x is not None else np.NaN for x in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)
