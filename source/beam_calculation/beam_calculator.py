#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:26:52 2023.

@author: placais
"""
from typing import Any, Callable
from abc import ABC, abstractmethod

import numpy as np

from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings
from core.elements import _Element
from core.list_of_elements import ListOfElements
from util.helper import recursive_items, recursive_getter


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
    def set_absolute_meshes(self, pos_in: float, s_in: int
                            ) -> tuple[float, int]:
        """Set the absolute indexes and arrays, depending on previous elem."""

    @abstractmethod
    def re_set_for_broken_cavity(self):
        """Modify solver parameters once a cavity is non-accelerating."""

    @abstractmethod
    def transf_mat_function_wrapper(self, w_kin_in: float, *args, **kwargs
                                    ) -> dict:
        """Calculate the beam propagation in _Element, give results dict."""

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            **kwargs: dict) -> Any:
        """Shorthand to get attributes."""
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]
        if len(out) == 1:
            return out[0]

        return tuple(out)
