#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:27:20 2023.

@author: placais

This module holds the class `Constraint`, which stores a constraint with its
name, limits, and methods to evaluate if it is violated or not.

"""
import logging
from dataclasses import dataclass

import numpy as np

from core.list_of_elements import equiv_elt

from beam_calculation.output import SimulationOutput

from util.dicts_output import markdown


IMPLEMENTED = ('phi_s')


@dataclass
class Constraint:
    """
    A single constraint.

    For now, it can only be a synchronous phase limits.

    """

    name: str
    cavity_name: str
    limits: tuple

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        self.limits_fmt = self.limits
        if 'phi' in self.name:
            self.limits_fmt = np.rad2deg(self.limits)

        if self.name not in IMPLEMENTED:
            logging.warning("Constraint not tested.")

    def __str__(self) -> str:
        """Output constraint name and limits."""
        out = f"{markdown[self.name]:20} {self.cavity_name:15}      "
        out += f"limits={self.limits_fmt[0]:>8.3f} {self.limits_fmt[1]:>8.3f}"
        return out

    def get_value(self, simulation_output: SimulationOutput, **kwargs: bool
                  ) -> float:
        """Get from the `SimulationOutput the quantity called `self.name`."""
        elt = equiv_elt(simulation_output.elts, self.cavity_name)
        return elt.get(self.name, **kwargs)

    def evaluate(self, simulation_output: SimulationOutput, **kwargs
                 ) -> tuple[float, float]:
        """Check if constraint is respected."""
        value = self.get_value(simulation_output, **kwargs)
        const = (value - self.limits[0],
                 self.limits[1] - value)
        return const
