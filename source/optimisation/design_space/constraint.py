#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:27:20 2023.

@author: placais

This module holds :class:`Constraint`, which stores a constraint with its
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
        # in particular: phi_s is hard-coded in get_value!!

        self._to_deg = False
        self._to_numpy = False

    def __str__(self) -> str:
        """Output constraint name and limits."""
        out = f"{markdown[self.name]:25} | {self.cavity_name:15} | {' ':<8} | "
        out += f"limits={self.limits_fmt[0]:>9.3f} {self.limits_fmt[1]:>9.3f}"
        return out

    @staticmethod
    def str_header() -> str:
        """Give information on what :func:`__str__` is about."""
        header = f"{'Variable':<25} | {'Element':<15} | {'x_0':<8} | "
        header += f"{'Lower lim':<9} | {'Upper lim':<9}"
        return header

    @property
    def kwargs(self) -> dict[str, bool]:
        """Return the `kwargs` to send a `get` method."""
        _kwargs = {'to_deg': self._to_deg,
                   'to_numpy': self._to_numpy}
        return _kwargs

    @property
    def n_constraints(self) -> int:
        """
        Return number of embedded constraints in this object.

        A lower + and upper bound count as two constraints.

        """
        return np.where(~np.isnan(np.array(self.limits)))[0].shape[0]

    def get_value(self, simulation_output: SimulationOutput) -> float:
        """Get from the `SimulationOutput` the quantity called `self.name`."""
        return simulation_output.get(self.name, **self.kwargs)

    def evaluate(self, simulation_output: SimulationOutput
                 ) -> tuple[float, float]:
        """Check if constraint is respected. They should be < 0."""
        value = self.get_value(simulation_output)
        const = (self.limits[0] - value,
                 value - self.limits[1])
        return const
