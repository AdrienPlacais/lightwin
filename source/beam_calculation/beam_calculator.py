#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:26:52 2023.

@author: placais
"""
from typing import Any, Callable
from abc import ABC, abstractmethod

from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings
from core.elements import _Element
from core.list_of_elements import ListOfElements


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
        return self.run_with_this(None, elts)

    @abstractmethod
    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements) -> SimulationOutput:
        """
        Perform a simulation with new cavity settings.

        Calling it with set_of_cavity_settings = None should be the same as
        calling the plain `run` method.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
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
    def _generate_element_to_index_func(self, elts: ListOfElements
                                        ) -> Callable[[_Element, str | None],
                                                      int | slice]:
        """Create the func to easily get data at proper mesh index."""

    @abstractmethod
    def _format_this(self, set_of_cavity_settings: SetOfCavitySettings) -> Any:
        """Transform `set_of_cavity_settings` for this BeamCalculator."""


class SingleElementCalculatorParameters(ABC):
    """
    Parent class to hold solving parameters. Attribute of _Element.

    As for now, only used by Envelope1D. Useful for TraceWin? To store its
    meshing maybe?
    """

    @abstractmethod
    def transf_mat_function_wrapper(self, w_kin_in: float, *args, **kwargs
                                    ) -> dict:
        """Calculate the beam propagation in _Element, give results dict."""
