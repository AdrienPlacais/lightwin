#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:47:04 2023.

@author: placais
"""
from dataclasses import dataclass
import logging


@dataclass
class SetOfCavitySettings(dict):
    """Holds cavity settings, to be tried during optimisation process."""


@dataclass
class SingleCavitySettings:
    """Settings of a single cavity."""

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


michel = SingleCavitySettings(k_e=0.8, phi_s=None, phi_0_rel=14.)
print(michel)
