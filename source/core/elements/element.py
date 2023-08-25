#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds :class:`Element`, the base parent class that is then declined
in Drift, FieldMap, etc.

.. todo::
    check FLAG_PHI_S_FIT

:: todo::
    rf_param should also return phi_rf_rel. Will be necessary for non-synch
    particles.

:: todo::
    __repr__ won't work with retuned elements

"""
from typing import Any
import numpy as np

from core.electric_field import RfField

from util.helper import recursive_items, recursive_getter

from failures.set_of_cavity_settings import SingleCavitySettings

from beam_calculation.single_element_beam_calculator_parameters import (
   SingleElementCalculatorParameters)


class Element():
    """Generic element."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """
        Init parameters common to all elements.

        Parameters
        ----------
        line : list of string
            A valid line of the ``.dat`` file.

        """
        self.line = line
        self.elt_info = {
            'elt_name': None,
            'nature': line[0],
            'status': 'none',    # Only make sense for cavities
        }
        self.length_m = 1e-3 * float(line[1])

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.acc_field = RfField()

        self.idx = {'dat_idx': dat_idx,
                    'elt_idx': None,
                    'lattice': None,
                    'section': None}
        self.beam_calc_param: dict[str, SingleElementCalculatorParameters] = {}

    def __str__(self) -> str:
        return self.elt_info['elt_name']

    def __repr__(self) -> str:
        # if self.elt_info['status'] not in ['none', 'nominal']:
        #     logging.warning("Element properties where changed.")
        # return f"{self.__class__}(line={self.line})"
        return self.__str__()

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            **kwargs: bool | str | None) -> Any:
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
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def update_status(self, new_status: str) -> None:
        """
        Change the status of a cavity.

        We also ensure that the value new_status is correct. If the new value
        is 'failed', we also set the norm of the electric field to 0.

        """
        assert self.elt_info['nature'] == 'FIELD_MAP', 'The status of an ' + \
            'element only makes sense for cavities.'

        authorized_values = [
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
        ]
        assert new_status in authorized_values

        self.elt_info['status'] = new_status
        if new_status == 'failed':
            self.acc_field.k_e = 0.
            for beam_calc_param in self.beam_calc_param.values():
                beam_calc_param.re_set_for_broken_cavity()

    def keep_rf_field(self, rf_field: dict, v_cav_mv: float, phi_s: float,
                      ) -> None:
        """Save data calculated by :func:`BeamCalculator.run_with_this`."""
        if rf_field != {}:
            self.acc_field.v_cav_mv = v_cav_mv
            self.acc_field.phi_s = phi_s
            self.acc_field.phi_0['phi_0_abs'] = rf_field['phi_0_abs']
            self.acc_field.phi_0['phi_0_rel'] = rf_field['phi_0_rel']
            self.acc_field.k_e = rf_field['k_e']

    def rf_param(self, solver_id: str, phi_bunch_abs: float, w_kin_in: float,
                 cavity_settings: SingleCavitySettings | None = None,
                 ) -> dict:
        """
        Set the properties of the rf field; in the default case, returns None.

        Parameters
        ----------
        solver_id : str
        Identificator of the :class:`BeamCalculator`.
        phi_bunch_abs : float
            Absolute phase of the particle (bunch frequency).
        w_kin_in : float
            Kinetic energy at the Element entrance in MeV.
        cavity_settings : SingleCavitySettings | None, optional
            Cavity settings. Should be None in a non-accelerating element such
            as a Drift or a broken FieldMap, and in accelerating elements
            outside the fit process. The default is None.

        Returns
        -------
        rf_parameters : dict
            Always {} by default.

        """
        return {}

    def is_accelerating(self) -> bool:
        """Say if this element is accelerating or not."""
        return self.get('nature') == 'FIELD_MAP' \
            and self.get('status') != 'failed'
