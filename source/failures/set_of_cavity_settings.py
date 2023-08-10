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
    _tracewin_command: list[str] | None = None

    def __post_init__(self):
        """Test that only one phase was given."""
        if not self._is_valid_phase_input():
            logging.error("You gave SingleCavitySettings several phases... "
                          "Which one should it take? Ignoring phases.")
            self.phi_0_abs = None
            self.phi_0_rel = None
            self.phi_s = None

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
