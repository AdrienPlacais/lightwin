#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:56 2023.

@author: placais

This module holds :class:`Objective`, a generic object to hold an optimisation
problem objective.

"""
import logging

from core.elements.element import Element
from core.beam_parameters import mismatch_from_arrays

from beam_calculation.output import SimulationOutput


class Objective:
    """
    Holds an objective, its ideal value, methods to evaluate it.

    """

    def __init__(self,
                 name: str,
                 scale: float,
                 element: Element | str,
                 pos: str,
                 reference_simulation_output: SimulationOutput | None = None,
                 reference_value: tuple[float] | float | None = None) -> None:
        """Set complementary :func:`get` flags, reference value."""
        self.name = name
        self.scale = scale
        self.element = element
        self.pos = pos
        self._to_deg = False
        self._to_numpy = False

        if 'mismatch_factor' in self.name:
            logging.warning("Some dirty patches installed to make mismatch "
                            "work. FIXME")
            # 1: manual edit of .get_value method
            # 2: manual edit of .evaluate method
            # 3: manual edit of .ref property
            # 4: manual edit of .this method

        if reference_simulation_output is not None:
            self.reference_value = self.get_value(reference_simulation_output)
            if reference_value is not None:
                logging.warning("You must provide :class:`Objective` a "
                                "reference simulation output or value. Using "
                                "reference simulation output...")
            return
        if reference_value is not None:
            self.reference_value = reference_value
            return

        logging.error("You must provide :class:`Objective` a reference (ideal "
                      " value or a reference :class:`SimulationOutput`. "
                      "Setting reference value to 0.")
        self.reference_value = 0.

    def __str__(self) -> str:
        """Give objective information value."""
        message = f"{self.name:>23} @elt {str(self.element):>5} "
        message += f"({self.pos:>3}) | {self.scale:>5} | "
        return message

    @property
    def ref(self) -> str:
        """Give `self.__str__` but with objective value of objective."""
        if 'mismatch_factor' in self.name:
            return str(self) + f"{self.reference_value}"
        if isinstance(self.reference_value, tuple):
            return str(self) + f"within {self.reference_value}"
        return str(self) + f"{self.reference_value:>10}"

    def this(self, simulation_output: SimulationOutput) -> str:
        """Give `self.__str__` but with evaluation of objective."""
        value = self.get_value(simulation_output)
        residue = self.evaluate(value)
        if 'mismatch_factor' in self.name:
            message = str(self) + f"{self.value} -> residue: {residue:>10}"
            return message
        message = str(self) + f"{self.value:>10} -> residue: {residue:>10}"
        return message

    @property
    def kwargs(self) -> dict[str, Element | str | bool]:
        """Return the `kwargs` to send a `get` method."""
        _kwargs = {'elt': self.element,
                   'pos': self.pos,
                   'to_deg': self._to_deg,
                   'to_numpy': self._to_numpy}
        return _kwargs

    def get_value(self, simulation_output: SimulationOutput) -> float:
        """Get from the `SimulationOutput` the quantity called `self.name`."""
        if 'mismatch_factor' in self.name:
            value = simulation_output.get('twiss_zdelta',
                                          elt=self.element,
                                          pos=self.pos,
                                          to_numpy=True)
            return value
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
            value = self.get_value(simulation_output)

        if 'mismatch_factor' in self.name:
            return mismatch_from_arrays(self.reference_value,
                                        value)[0] * self.scale
        if isinstance(self.reference_value, tuple):
            if value < self.reference_value[0]:
                return self.scale * (value - self.reference_value[0])**2
            if value > self.reference_value[1]:
                return self.scale * (value - self.reference_value[1])**2
            return 0.
        return self.scale * (value - self.reference_value)
