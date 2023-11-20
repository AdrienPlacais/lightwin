#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds :class:`Element`, declined in Drift, FieldMap, etc.

.. todo::
    check FLAG_PHI_S_FIT

.. todo::
    rf_param should also return phi_rf_rel. Will be necessary for non-synch
    particles.

.. todo::
    __repr__ won't work with retuned elements

"""
from typing import Any
import numpy as np

from core.instruction import Instruction
from core.electric_field import RfField

from util.helper import recursive_items, recursive_getter

from failures.set_of_cavity_settings import SingleCavitySettings

from beam_calculation.single_element_beam_calculator_parameters import (
    SingleElementCalculatorParameters)


class Element(Instruction):
    """Generic element."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 elt_name: str | None = None,
                 **kwargs: str) -> None:
        """
        Init parameters common to all elements.

        Parameters
        ----------
        line : list[str]
            A line of the ``.dat`` file. If the element was given a name, it
            must not appear in ``line`` but rather in ``elt_name``. First
            element of the list must be in :data:`.IMPLEMENTED_ELEMENTS`.
        dat_idx : int
            Position in the ``.dat`` file.
        elt_name : str | None, optional
            Non-default name of the element, as given in the ``.dat`` file. The
            default is None, in which case an automatic name will be given
            later.

        """
        super().__init__(line, dat_idx, is_implemented=True)

        self.elt_info = {
            'elt_name': elt_name,
            'nature': line[0],
            'status': 'none',    # Only make sense for cavities
        }
        self.length_m = 1e-3 * float(line[1])

        # By default, an element is non accelerating and has a dummy
        # accelerating field.
        self.acc_field = RfField()

        new_idx = {'elt_idx': None,
                   'increment_elt_idx': True,
                   'lattice': None,
                   'idx_in_lattice': None,
                   'increment_lattice_idx': True,
                   'section': None,
                   }
        self.idx = self.idx | new_idx
        self.beam_calc_param: dict[str, SingleElementCalculatorParameters] = {}

    def __str__(self) -> str:
        """Give the same name as TraceWin would."""
        out = self.elt_info['elt_name']
        if out is None:
            out = str(self.line)
        return out

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

    @property
    def is_accelerating(self) -> bool:
        """Say if this element is accelerating or not.

        Will return False by default.

        """
        return False

    @property
    def can_be_retuned(self) -> bool:
        """Tell if we can modify the element's tuning.

        Will return False by default.

        """
        return False
