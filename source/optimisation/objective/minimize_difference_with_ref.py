#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:56 2023.

@author: placais

This module holds an objective that is a simple difference over a given
quantity between the reference linac and the linac under tuning.

"""
import logging

from optimisation.objective.objective import Objective

from core.elements.element import Element

from beam_calculation.output import SimulationOutput


class MinimizeDifferenceWithRef(Objective):
    """A simple difference at a given point between ref and fix."""

    def __init__(self,
                 name: str,
                 weight: float,
                 get_key: str,
                 get_kwargs: dict[str, Element | str | bool],
                 reference: SimulationOutput,
                 descriptor: str | None = None,
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
        reference : SimulationOutput
            The reference simulation output from which the ideal value will be
            taken.

        """
        self.get_key = get_key
        self.get_kwargs = get_kwargs
        super().__init__(name,
                         weight,
                         descriptor=descriptor,
                         ideal_value=self._value_getter(reference))
        self._check_ideal_value()

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
        if isinstance(self.ideal_value, float):
            message += f"{self.ideal_value:>10}"
            return message
        message += f"{self.ideal_value}"
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

    def _check_ideal_value(self) -> None:
        """Assert the the reference value is a float."""
        if not isinstance(self.ideal_value, float):
            logging.warning(f"Tried to get {self.get_key} with "
                            f"{self.get_kwargs}, which returned "
                            f"{self.ideal_value} instead of a float.")

    def evaluate(self, simulation_output: SimulationOutput) -> float:
        value = self._value_getter(simulation_output)
        return self._compute_residues(value)

    def _compute_residues(self, value: float) -> float:
        """Compute the residues."""
        return self.weight * (value - self.ideal_value)
