#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds classes to store (compensating) cavity settings."""
from dataclasses import dataclass
import logging
from typing import Any, TypeVar
import numpy as np

from tracewin_utils.interface import single_cavity_settings_to_command

from core.electric_field import phi_0_abs_with_new_phase_reference

from util.helper import recursive_items, recursive_getter


FieldMap = TypeVar('FieldMap')


@dataclass
class SingleCavitySettings:
    """Settings of a single cavity."""

    cavity: FieldMap
    k_e: float | None = None
    phi_0_abs: float | None = None
    phi_0_rel: float | None = None
    phi_s: float | None = None
    index: int | None = None

    def __post_init__(self):
        """Test that only one phase was given. Set `_tracewin_command attr`."""
        if not self._is_valid_phase_input:
            logging.error("You gave SingleCavitySettings several phases... "
                          "Which one should it take? Ignoring phases.")
            self.phi_0_abs = None
            self.phi_0_rel = None
            self.phi_s = None

    def re_set_element_index_to_absolute_value(self) -> None:
        """
        Change cavity index to its position in full linac.

        When switching from a sub-`ListOfElements` during the optimisation
        process to the full `ListOfElements` after the optimisation, we must
        update index `n` in the `ele[n][v]]` command.

        """
        self.index = self.cavity.get('elt_idx', to_numpy=False)

    def tracewin_command(self, delta_phi_bunch: float = 0.) -> list[str]:
        """Call the function from `tracewin_utils` to modify TraceWin call."""
        phi_0_abs = self._tracewin_phi_0_abs(delta_phi_bunch)
        tracewin_command = single_cavity_settings_to_command(self.index,
                                                             phi_0_abs,
                                                             self.k_e)
        return tracewin_command

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_deg: bool = False, **kwargs: dict
            ) -> tuple[Any]:
        """Shorthand to get attributes."""
        val: dict[str, Any] = {}
        for key in keys:
            val[key] = []

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

    @property
    def _is_valid_phase_input(self) -> bool:
        """Assert that no more than one phase was given as input."""
        phases = [self.phi_0_abs, self.phi_0_rel, self.phi_s]
        number_of_given_phases = sum(1 for phase in phases
                                     if phase is not None)
        if number_of_given_phases > 1:
            return False
        return True

    def _tracewin_phi_0_abs(self, delta_phi_bunch: float) -> float:
        """
        Return the proper absolute entry phase of the cavity.

        The attribute `self.phi_0_abs` is only valid if the beam has a null
        absolute phase at the entry of the linac. When working with `TraceWin`,
        the beam has an absolute phase at the entry of the compensation zone.
        Hence we compute the new `phi_0_abs` that will be properly taken into
        account.

        Three figure cases, according to the nature of the input:
            - phi_0_rel
            - phi_0_abs (phi_abs = 0. at entry of `ListOfElements`)
            - phi_s

        Parameters
        ----------
        delta_phi_bunch : float
            Difference between the absolute entry phase of the `ListOfElements`
            under study and the entry phase of the `ListOfElements` for which
            given phi_0_abs is valid.

        Returns
        -------
        phi_0 : float
            New absolute phase.

        """
        if not self._is_valid_phase_input:
            logging.error("More than one phase was given.")

        if self.phi_0_abs is not None:
            phi_0_abs = phi_0_abs_with_new_phase_reference(
                self.phi_0_abs,
                delta_phi_bunch * self.cavity.acc_field.n_cell)
            return phi_0_abs

        if self.phi_0_rel is not None:
            raise NotImplementedError

        if self.phi_s is not None:
            raise NotImplementedError


@dataclass
class SetOfCavitySettings(dict[FieldMap, SingleCavitySettings]):
    """
    Holds several cavity settings, to be tried during optimisation process.

    """

    __cavity_settings: list[SingleCavitySettings]

    def __post_init__(self):
        """Create the proper dictionary."""
        my_set = {single_setting.cavity: single_setting
                  for single_setting in self.__cavity_settings}
        super().__init__(my_set)

    def tracewin_command(self, delta_phi_bunch: float = 0.) -> list[str]:
        """Set TraceWin command modifier for current settings."""
        _tracewin_command = []
        for settings in self.values():
            _tracewin_command.extend(
                settings.tracewin_command(delta_phi_bunch=delta_phi_bunch)
            )
        return _tracewin_command

    def re_set_elements_index_to_absolute_value(self) -> None:
        """Update cavities index to properly set `ele[n][v]` commands."""
        for setting in self.values():
            setting.re_set_element_index_to_absolute_value()
