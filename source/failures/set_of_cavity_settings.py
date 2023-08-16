#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:47:04 2023.

@author: placais

This module holds classes to store (compensating) cavity settings in a compact
way.

"""
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
        if not self._is_valid_phase_input():
            logging.error("You gave SingleCavitySettings several phases... "
                          "Which one should it take? Ignoring phases.")
            self.phi_0_abs = None
            self.phi_0_rel = None
            self.phi_s = None
        self._tracewin_command: list[str] | None = None

    def update_to_full_list_of_elements(self,
                                        delta_phi_rf: float,
                                        ) -> None:
        """
        Rephase the cavity, change its index.

        When switching from a sub-`ListOfElements` during the optimisation
        process to the full `ListOfElements` after the optimisation, we must
        take care of two things when using TraceWin:
            - index `n` in the `ele[n][v]]` command must be updated;
            - abs phase must de-de-phased and expressed w.r.t. to the absolute
            phase of the full `ListOfElements` (not the absolute phase of the
            sub-`ListOfElements`).

        If we made a simulation with relative phases, we have nothing to
        change.

        """
        self.index = self.cavity.get('elt_idx', to_numpy=False)

        if self.phi_0_abs is not None:
            self.phi_0_abs = phi_0_abs_with_new_phase_reference(self.phi_0_abs,
                                                                delta_phi_rf)

    @property
    def tracewin_command(self):
        """Call the function from `tracewin_utils` to modify TraceWin call."""
        if self._tracewin_command is None:
            abs_flag = None
            if self.phi_0_rel is not None:
                logging.warning("Relative phase in command line for TW not "
                                "validated yet.")
                abs_flag = 0
            phi = next(phase for phase in [self.phi_0_abs, self.phi_0_rel,
                                           self.phi_s]
                       if phase is not None)
            self._tracewin_command = single_cavity_settings_to_command(
                self.index,
                phi,
                self.k_e,
                abs_flag
            )
            if self.phi_s is not None:
                logging.error("Synchronous phase in command line for TW not "
                              "implemented.")
        return self._tracewin_command

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

    def _is_valid_phase_input(self) -> bool:
        """Assert that no more than one phase was given as input."""
        phases = [self.phi_0_abs, self.phi_0_rel, self.phi_s]
        number_of_given_phases = sum(1 for phase in phases
                                     if phase is not None)
        if number_of_given_phases > 1:
            return False
        return True


@dataclass
class SetOfCavitySettings(dict[FieldMap, SingleCavitySettings]):
    """
    Holds several cavity settings, to be tried during optimisation process.

    """

    __cavity_settings: list[SingleCavitySettings]
    _tracewin_command: list[str] | None = None

    def __post_init__(self):
        """Create the proper dictionary."""
        my_set = {single_setting.cavity: single_setting
                  for single_setting in self.__cavity_settings}
        super().__init__(my_set)

    @property
    def tracewin_command(self):
        """Set TraceWin command modifier for current settings."""
        if self._tracewin_command is None:
            self._tracewin_command = []
            for settings in self.values():
                self._tracewin_command.extend(settings.tracewin_command)
        return self._tracewin_command

    def update_to_full_list_of_elements(self) -> None:
        """Update all the `SingleCavitySettings` after optimisation with TW."""
        for cavity, setting in self.items():
            new_phi_in = 0. * cavity.acc_field.n_cell
            old_phi_in = cavity.acc_field.phi_0['new_reference_phase']
            delta_phi_rf = new_phi_in - old_phi_in
            setting.update_to_full_list_of_elements(delta_phi_rf)
