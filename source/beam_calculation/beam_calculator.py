#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:26:52 2023.

@author: placais
"""
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass

from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings

@dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    @abstractmethod
    def run(self) -> SimulationOutput:
        """Perform a simulation with default settings."""

    @abstractmethod
    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings
                      ) -> SimulationOutput:
        """Perform a simulation with new cavity settings."""

    @abstractmethod
    def _generate_simulation_output(self) -> SimulationOutput:
        """Transform the outputs of BeamCalculator to a SimulationOutput."""

    @abstractmethod
    def _format_this(self, set_of_cavity_settings: SetOfCavitySettings) -> Any:
        """Transform `set_of_cavity_settings` for this BeamCalculator."""
