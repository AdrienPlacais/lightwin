#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds :class:`Constraint`, which stores a constraint.

It saves it's name, limits, and methods to evaluate if it is violated or not.

"""
import logging
from dataclasses import dataclass

import numpy as np

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from optimisation.design_space.design_space_parameter import \
    DesignSpaceParameter


IMPLEMENTED_CONSTRAINTS = ('phi_s',)  #:


@dataclass
class Constraint(DesignSpaceParameter):
    """
    A single constraint.

    For now, it can only be a synchronous phase limits.

    """

    def __post_init__(self):
        """Convert values in deg for output if it is angle."""
        if self.name not in IMPLEMENTED_CONSTRAINTS:
            logging.warning("Constraint not tested.")
        # in particular: phi_s is hard-coded in get_value!!

        super().__post_init__()
        self._to_deg = False
        self._to_numpy = False

    @property
    def kwargs(self) -> dict[str, bool]:
        """Return the `kwargs` to send a `get` method."""
        _kwargs = {'to_deg': self._to_deg,
                   'to_numpy': self._to_numpy,
                   'elt': self.element_name}
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
