#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:26:52 2023.

@author: placais
"""
from typing import Any
from abc import ABC, abstractmethod
# from dataclasses import dataclass

from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings
from core.list_of_elements import ListOfElements

# @dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    @abstractmethod
    def run(self, elts: ListOfElements) -> SimulationOutput:
        """
        Perform a simulation with default settings.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """

    @abstractmethod
    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings,
                      elts: ListOfElements) -> SimulationOutput:
        """
        Perform a simulation with new cavity settings.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings
            Holds the norms and phases of the compensating cavities.
        elts: ListOfElements
            List of elements in which the beam should be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """

    @abstractmethod
    def _generate_simulation_output(self, *args: Any) -> SimulationOutput:
        """Transform the outputs of BeamCalculator to a SimulationOutput."""

    @abstractmethod
    def _format_this(self, set_of_cavity_settings: SetOfCavitySettings) -> Any:
        """Transform `set_of_cavity_settings` for this BeamCalculator."""
