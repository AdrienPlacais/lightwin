#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 15:47:04 2023.

@author: placais
"""
from dataclasses import dataclass
import logging
from typing import Any, Optional
import numpy as np

from util.helper import recursive_items, recursive_getter


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
class SetOfCavitySettings(dict[object, SingleCavitySettings]):
    """Holds cavity settings, to be tried during optimisation process."""
    __cavity_settings: list[SingleCavitySettings]

    def __post_init__(self):
        """Create the proper dictionary."""
        my_set = {single_setting.cavity: single_setting
                  for single_setting in self.__cavity_settings}
        super().__init__(my_set)
