#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold parameters that are shared by all cavities of same type.

See Also
--------
CavitySettings

"""
import logging
from typing import Any, Callable

import cmath
import numpy as np

import config_manager as con
from util.helper import recursive_items, recursive_getter


def compute_param_cav(integrated_field: complex) -> dict[str, float]:
    """Compute synchronous phase and accelerating field."""
    polar_itg = cmath.polar(integrated_field)
    cav_params = {'v_cav_mv': polar_itg[0],
                  'phi_s': polar_itg[1]}
    return cav_params


class RfField():
    r"""Cos-like RF field.

    .. deprecated:: 0.6.16
        Will be separated into :class:`NewRfField` for parameters specific to
        cavity design and :class:`.CavitySettings` for parameters specific to
        the cavity under study.

    Warning, all phases are defined as:

    .. math::
        \phi = \omega_0^{rf} t

    While in the rest of the code it is defined as:

    .. math::
        \phi = \omega_0_^{bunch} t

    All phases are stored in radian.

    Attributes
    ----------
    e_spat : Callable[[float], float]
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
            :class:`.ListOfElements` under study does not start at the
            beginning of the linac and we use TraceWin.
            new_reference_phase : phase at the entrance of this
            `ListOfElements`.
    v_cav_mv : float
        Cavity accelerating field in MV.
    phi_s : float
        Cavity synchronous phase in rad.
    omega0_rf : float
        RF pulsation of the cavity in rad/s.
    bunch_to_rf : float
        :math:`f_{rf} / f_{bunch}`. In particular, it is used to convert the rf
        absolute phase given by the transfer matrix function of
        :class:`.Envelope1D` and :class:`.Envelope3D` to bunch absolute phase.
    n_cell : int
        Number of cells in the cavity.
    n_z : int | None
        Number of points in the file that gives `e_spat`, the spatial component
        of the electric field.

    """

    def __init__(self,
                 k_e: float = np.NaN,
                 absolute_phase_flag: bool = False,
                 phi_0: float | None = None) -> None:
        """Instantiate object."""
        self.e_spat: Callable[[float], float]
        self.n_cell: int
        self.set_e_spat(lambda _: 0., n_cell=2)

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

        # Default values, overwritten by the FREQ command
        self.omega0_rf: float
        self.bunch_to_rf: float
        self.set_rf_freq(con.F_BUNCH_MHZ)

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

    def set_rf_freq(self,
                    f_mhz: float,
                    ) -> None:
        """Initialize the pulsation and the rf / bunch fraction."""
        self.omega0_rf = 2e6 * np.pi * f_mhz
        self.bunch_to_rf = f_mhz / con.F_BUNCH_MHZ

    def set_e_spat(self,
                   e_spat: Callable[[float], float],
                   n_cell: int) -> None:
        """Set the pos. component of electric field, set number of cells."""
        self.e_spat = e_spat
        self.n_cell = n_cell

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
            delta_phi_rf *= self.bunch_to_rf
            new_phi_in *= self.bunch_to_rf

        new_phi_0_abs = phi_0_abs_with_new_phase_reference(
            self.phi_0['phi_0_abs'],
            delta_phi_rf
        )

        self.phi_0['new_reference_phase'] = new_phi_in
        self.phi_0['phi_0_abs_but_reference_phase_is_different'] = \
            new_phi_0_abs
        return new_phi_0_abs


class NewRfField():
    r"""Cos-like RF field.

    Warning, all phases are defined as:

    .. math::
        \phi = \omega_0^{rf} t

    While in the rest of the code it is defined as:

    .. math::
        \phi = \omega_0_^{bunch} t

    All phases are stored in radian.

    Attributes
    ----------
    e_spat : Callable[[float], float]
        Spatial component of the electric field. Needs to be multiplied by the
        cos(omega t) to have the full electric field. Initialized to null
        function.
    n_cell : int
        Number of cells in the cavity.
    n_z : int | None
        Number of points in the file that gives `e_spat`, the spatial component
        of the electric field.

    """

    def __init__(self) -> None:
        """Instantiate object."""
        self.e_spat: Callable[[float], float]
        self.n_cell: int
        self.n_z: int

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return hasattr(self, key)

    def get(self,
            *keys: str,
            **kwargs: bool | str | None
            ) -> Any:
        """Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        **kwargs : bool | str | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val: dict[str, Any] = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = getattr(self, key)

        out = [val[key] for key in keys]
        if len(keys) == 1:
            return out[0]
        return tuple(out)

    def set_e_spat(self,
                   e_spat: Callable[[float], float],
                   n_cell: int) -> None:
        """Set the pos. component of electric field, set number of cells."""
        self.e_spat = e_spat
        self.n_cell = n_cell
