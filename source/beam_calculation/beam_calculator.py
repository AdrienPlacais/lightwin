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
from dataclasses import dataclass
from abc import ABC, abstractmethod

import config_manager as con
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.list_of_elements.list_of_elements import ListOfElements
from core.accelerator import Accelerator

from beam_calculation.simulation_output.factory import SimulationOutputFactory
from core.list_of_elements.factory import ListOfElementsFactory


@dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    out_folder: str

    def __post_init__(self):
        """Set ``id`` and factories."""
        self.id: str = self.__repr__()
        self.simulation_output_factory: SimulationOutputFactory
        self.list_of_elements_factory: ListOfElementsFactory
        self._set_up_common_factories()
        self._set_up_specific_factories()

    def _set_up_common_factories(self) -> None:
        """Create the factories declared in :meth:`__post_init__`."""
        self.list_of_elements_factory = ListOfElementsFactory(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation,
            con.F_BUNCH_MHZ,
            default_field_map_folder='/home/placais/LightWin/data',
        )

    @abstractmethod
    def _set_up_specific_factories(self) -> None:
        """Set up the factories specific to the :class:`.BeamCalculator`."""

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
