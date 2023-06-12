#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:47:04 2023.

@author: placais
"""
from dataclasses import dataclass
import logging


@dataclass
class SingleCavitySettings:
    """Settings of a single cavity."""

    cavity: object
    k_e: float | None = None
    phi_0_abs: float | None = None
    phi_0_rel: float | None = None
    phi_s: float | None = None

    def __post_init__(self):
        """Test that only one phase was given."""
        if not self._is_valid_phase_input():
            logging.error("You gave SingleCavitySettings several phases... "
                          "Which one should it take? Ignoring phases.")
            self.phi_0_abs = None
            self.phi_0_rel = None
            self.phi_s = None

    def _is_valid_phase_input(self) -> bool:
        """Assert that no more than one phase was given as input."""
        phases = [self.phi_0_abs, self.phi_0_rel, self.phi_s]
        number_of_given_phases = sum(1 for phase in phases
                                     if phase is not None)
        if number_of_given_phases > 1:
            return False
        return True

@dataclass
class SetOfCavitySettings(dict[object, SingleCavitySettings]):
    """Holds cavity settings, to be tried during optimisation process."""
    __cavity_settings: list[SingleCavitySettings]

    def __post_init__(self):
        """Create the proper dictionary."""
        my_set = {single_setting.cavity: single_setting
                  for single_setting in self.__cavity_settings}
        super().__init__(my_set)

