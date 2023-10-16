#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:56 2023.

@author: placais

This module holds an objective that is a quantity must be within some bounds.

.. todo::
    Implement loss functions.

"""
import logging

from optimisation.objective.objective import Objective

from core.elements.element import Element

from beam_calculation.output import SimulationOutput


class QuantityIsBetween(Objective):
    """Quantity must be within some bounds."""

    def __init__(self,
                 name: str,
                 weight: float,
                 get_key: str,
                 get_kwargs: dict[str, Element | str | bool],
                 limits: tuple[float],
                 descriptor: str | None = None,
                 loss_function: str | None = None
                 ) -> None:
        """
        Set complementary :func:`get` flags, reference value.

        Parameters
        ----------
        get_key : str
            Name of the quantity to get, which must be an attribute of
            :class:`SimulationOutput`.
        get_kwargs : dict[str, Element | str | bool]
            Keyword arguments for the :func:`get` method. We do not check its
            validity, but in general you will want to define the keys ``elt``
            and ``pos``. If objective concerns a phase, you may want to precise
            the ``to_deg`` key. You also should explicit the ``to_numpy`` key.
        limits : tuple[float]
            Lower and upper bound for the value.
        loss_function : str | None, optional
            Indicates how the residues are handled whe the quantity is outside
            the limits. The default is None.

        """
        self.get_key = get_key
        self.get_kwargs = get_kwargs
        super().__init__(name,
                         weight,
                         descriptor=descriptor,
                         ideal_value=limits)
        if loss_function is not None:
            logging.warning("Loss functions not implemented.")

    def _base_str(self) -> str:
        """Return a base text for output."""
        message = f"{self.get_key:>23}"

        elt = str(self.get_kwargs.get('elt', 'NA'))
        message += f" @elt {elt:>5}"

        pos = str(self.get_kwargs.get('pos', 'NA'))
        message += f" ({pos:>3}) | {self.weight:>5} | "
        return message

    def __str__(self) -> str:
        """Give objective information value."""
        message = self._base_str()
        message += f"{self.ideal_value[0]:>4} | {self.ideal_value[1]:>4}"
        return message

    def current_value(self, simulation_output: SimulationOutput) -> str:
        value = self._value_getter(simulation_output)
        message = self._base_str()
        if isinstance(value, float):
            message += f"{value:>10} | {self._compute_residues(value):>10}"
            return message
        message += f"{value} | {self._compute_residues(value):>10}"
        return message

    def _value_getter(self, simulation_output: SimulationOutput
                      ) -> float:
        """Get desired value using :func:`SimulationOutput.get` method."""
        return simulation_output.get(self.get_key, **self.get_kwargs)

    def evaluate(self, simulation_output: SimulationOutput) -> float:
        value = self._value_getter(simulation_output)
        return self._compute_residues(value)

    def _compute_residues(self, value: float) -> float:
        """Compute the residues."""
        if value < self.ideal_value[0]:
            return self.weight * (value - self.ideal_value[0])**2
        if value > self.ideal_value[1]:
            return self.weight * (value - self.ideal_value[1])**2
        return 0.
