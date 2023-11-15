#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds a base class for beam propagation computing tools.

It defines the base class :class:`BeamCalculator`, which computes the
propagation of the beam in a :class:`.ListOfElements`, possibly with a specific
:class:`.SetOfCavitySettings` (optimisation process). It should return a
:class:`.SimulationOutput`.

.. todo::
    Precise that BeamParametersFactory and TransferMatrixFactory are mandatory.

"""
import logging
from typing import Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial
from beam_calculation.simulation_output.factory import SimulationOutputFactory

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.elements.element import Element
from core.list_of_elements.factory import ListOfElementsFactory
from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import equivalent_elt
from core.accelerator import Accelerator
from core.beam_parameters.factory import BeamParametersFactory
from core.transfer_matrix.factory import TransferMatrixFactory


@dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    out_folder: str

    def __post_init__(self):
        """Set ``id``."""
        self.id: str = self.__repr__()
        self.simulation_output_factory: SimulationOutputFactory
        self.list_of_elements_factory: ListOfElementsFactory
        self._set_up_factories()

    @abstractmethod
    def _set_up_factories(self) -> None:
        """Create the factories declared in :meth:`__post_init__`."""

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

        Calling it with `set_of_cavity_settings = None` should be the same as
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
    def post_optimisation_run_with_this(
        self,
        optimized_cavity_settings: SetOfCavitySettings | None,
        full_elts: ListOfElements
    ) -> SimulationOutput:
        """
        Run a simulation a simulation after optimisation is over.

        With `Envelope1D`, it just calls the classic `run_with_this`. But with
        TraceWin, we need to update the `optimized_cavity_settings` as running
        an optimisation run on a fraction of the linac is pretty different from
        running a simulation on the whole linac.

        """

    @abstractmethod
    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """Init some `BeamCalculator` solver parameters."""

    def _generate_simulation_output(self, *args, **kwargs) -> SimulationOutput:
        """Transform the output of ``run`` to a :class:`.SimulationOutput`."""
        return self.simulation_output_factory.run(*args, **kwargs)

    @property
    @abstractmethod
    def is_a_multiparticle_simulation(self) -> bool:
        """Tell if the simulation is a multiparticle simulation."""
        pass

    @property
    @abstractmethod
    def is_a_3d_simulation(self) -> bool:
        """Tell if the simulation is in 3D."""
        pass
