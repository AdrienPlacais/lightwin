#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 17:26:56 2023.

@author: placais

This module holds :class:`Objective`, an abstract base class from which every
objective will inherit.

"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from beam_calculation.output import SimulationOutput


class Objective(ABC):
    """
    Holds an objective and methods to evaluate it.

    Parameters
    ----------
    name : str
        A short string to describe the objective and access to it.
    weight : float
        A scaling constant to set the weight of current objective.
    descriptor : str | None, optional
        A longer string to explain the objective. The default is None.
    ideal_value : float | tuple[float], optional
        The ideal value or range of values that we should tend to.

    Methods
    -------
    evaluate : Callable[[SimulationOutput | float], float]
        A method that computes the residues over the current objective.
    current_value : Callable[[SimulationOutput | float], str]
        Give value of current objective and residue.

    """

    def __init__(self,
                 name: str,
                 weight: float,
                 descriptor: str | None = None,
                 ideal_value: float | tuple[float] = None
                 ) -> None:
        self.name = name
        self.weight = weight
        self.descriptor = descriptor
        self.ideal_value = ideal_value

    @abstractmethod
    def __str__(self) -> str:
        """Output info on what is this objective about."""

    def str_header(self) -> str:
        """Give a header to explain what :func:`__str__` returns."""
        header = "=*50" + "\n"
        header += f"{'What, where, etc':>40} | {'wgt.':>5} | "
        header += f"{'ideal val':>10}"
        header += "\n" + "-*50"
        return header

    @abstractmethod
    def current_value(self, simulation_output: SimulationOutput | float
                      ) -> str:
        """Give value of current objective and residue."""

    def current_value_header(self) -> str:
        """Give a header to explain what :func:`current_value` returns."""
        header = "=*50" + "\n"
        header += f"{'What, where, etc':>40} | {'wgt.':>5} | "
        header += f"{'current val':>10} | {'residue':>10}"
        header += "\n" + "-*50"
        return header

    @abstractmethod
    def evaluate(self, simulation_output: SimulationOutput | float) -> float:
        """Compute residue of this objective.

        Parameters
        ----------
        simulation_output : SimulationOutput | float
            Object containing simulation results of the broken linac.

        Returns
        -------
        residue : float
            Difference between current evaluation and ideal_value value for
            ``self.name``, scaled by ``self.weight``.

        """
