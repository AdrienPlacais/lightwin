#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:56 2023.

@author: placais

This module holds `Objective`, a generic object to hold an optimisation problem
objective.

"""
import logging
from dataclasses import dataclass

from core.elements import _Element

from beam_calculation.output import SimulationOutput


@dataclass
class Objective:
    """
    Holds an objective, its ideal value, methods to evaluate it.

    """

    def __init__(self,
                 name: str,
                 scale: float,
                 element: _Element | str,
                 pos: str,
                 reference_simulation_output: SimulationOutput | None = None,
                 reference_value: float | None = None) -> None:
        """Set complementary `get` flags, reference value."""
        self.name = name
        self.scale = scale
        self.element = element
        self.pos = pos
        self._to_deg = False
        self._to_numpy = False

        if reference_simulation_output is not None:
            self.reference_value = self.get_value(reference_simulation_output)
            if reference_value is not None:
                logging.warning("You must provide `Objective` a reference "
                                "simulation output or value. Using reference "
                                "simulation output...")
            return
        if reference_value is not None:
            self.reference_value = reference_value
            return

        logging.error("You must provide `Objective` a reference (ideal) value "
                      "or a reference `SimulationOutput`. Setting reference "
                      "value to 0.")
        self.reference_value = 0.

    def __str__(self) -> str:
        """Give objective information value."""
        message = f"{self.name:>10} @elt {self.element:>5} ({self.pos:>3}) | "
        message += f"{self.scale:>5} | "

    @property
    def ref(self) -> str:
        """Give `self.__str__` but with objective value of objective."""
        return str(self) + f"{self.reference_value:>10}"

    def this(self, simulation_output: SimulationOutput) -> str:
        """Give `self.__str__` but with evaluation of objective."""
        value = self.get_value(simulation_output)
        residue = self.evaluate(value)
        message = str(self) + f"{self.value:>10} -> residue: {residue:>10}"
        return message

    @property
    def kwargs(self) -> dict[str, _Element | str | bool]:
        """Return the `kwargs` to send a `get` method."""
        _kwargs = {'elt': self.element,
                   'pos': self.pos,
                   'to_deg': self._to_deg,
                   'to_numpy': self._to_numpy}
        return _kwargs

    def get_value(self, simulation_output: SimulationOutput) -> float:
        """Get from the `SimulationOutput` the quantity called `self.name`."""
        return simulation_output.get(self.name, **self.kwargs)

    def evaluate(self, simulation_output: SimulationOutput | float) -> float:
        """
        Compute residue of this objective.

        Parameters
        ----------
        simulation_output : SimulationOutput | float
            Object from which `self.name` should be got. If a float is
            provided, we use it instead.

        Returns
        -------
        residue : float
            Difference between current evaluation and reference value for
            `self.name`, scaled by `self.scale`.

        """
        value = simulation_output
        if isinstance(simulation_output, SimulationOutput):
            value = simulation_output.get_value(simulation_output)
        return self.scale * (value - self.reference_value)
