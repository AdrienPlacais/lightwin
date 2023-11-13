#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds class:`Variable`, which stores an optimisation variable.

It keeps it's name, bounds, initial value, etc.

"""
import logging
from dataclasses import dataclass

import numpy as np

from util.dicts_output import markdown


IMPLEMENTED = ('k_e', 'phi_0_abs', 'phi_0_rel', 'phi_s')


@dataclass
class Variable:
    """
    A single variable.

    It can be a cavity amplitude, absolute phase, relative phase or synchronous
    phase with an initial value and limits.

    """

    name: str
    element_name: str
    x_0: float
    limits: tuple

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        if self.name not in IMPLEMENTED:
            logging.warning(f"Variable {self.name} not tested.")

        self.x_0_fmt, self.limits_fmt = self.x_0, self.limits
        if 'phi' in self.name:
            self.x_0_fmt = np.rad2deg(self.x_0)
            self.limits_fmt = np.rad2deg(self.limits)

    def __str__(self) -> str:
        """Output variable name, initial value and limits."""
        out = f"{markdown[self.name]:25} | {self.element_name:15} | "
        out += f"{self.x_0_fmt:>8.3f} | "
        out += f"{self.limits_fmt[0]:>9.3f} | {self.limits_fmt[1]:>9.3f}"
        return out

    @staticmethod
    def str_header() -> str:
        """Give information on what :func:`__str__` is about."""
        header = f"{'Variable':<25} | {'Element':<15} | {'x_0':<8} | "
        header += f"{'Lower lim':<9} | {'Upper lim':<9}"
        return header
