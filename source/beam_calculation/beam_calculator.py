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
import time
import datetime
from dataclasses import dataclass
from abc import ABC, abstractmethod

import config_manager as con
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.list_of_elements.list_of_elements import ListOfElements
from core.accelerator.accelerator import Accelerator

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
        """
        Create the factories declared in :meth:`__post_init__`.

        .. todo::
            ``default_field_map_folder`` has a wrong default value. Should take
            path to the ``.dat`` file, that is not known at this point. Maybe
            handle this directly in the :class:`.InstructionsFactory` or
            whatever.

        """
        self.list_of_elements_factory = ListOfElementsFactory(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation,
            con.F_BUNCH_MHZ,
            default_field_map_folder='/home/placais/LightWin/data',
            load_field_maps=True,  # useless with TraceWin
            field_maps_in_3d=False,  # not implemented anyway
            # Different loading of field maps if Cython
            load_cython_field_maps=con.FLAG_CYTHON,
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

    def compute(self,
                accelerator: Accelerator,
                keep_settings: bool = True,
                recompute_reference: bool = True,
                output_time: bool = True,
                **kwargs: SimulationOutput,
                ) -> None:
        """Wrap full process to compute propagation of beam in accelerator.

        Parameters
        ----------
        accelerator : Accelerator
            Accelerator under study.
        keep_settings : bool, optional
            If settings/simulation output should be saved. The default is True.
        recompute_reference : bool, optional
            If results should be taken from a file instead of recomputing
            everything each time. The default is True.
        output_time : bool, optional
            To print in log the time the calculation took. The default is True.
        kwargs : SimulationOutput
            For calculation of mismatch factors..

        """
        start_time = time.monotonic()

        self.init_solver_parameters(accelerator)

        simulation_output = self.run(accelerator.elts)
        simulation_output.compute_complementary_data(accelerator.elts,
                                                     **kwargs)
        if keep_settings:
            accelerator.keep_settings(simulation_output)
            accelerator.keep_simulation_output(simulation_output,
                                               self.id)

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        if output_time:
            logging.info(f"Elapsed time in beam calculation: {delta_t}")

        if not recompute_reference:
            raise NotImplementedError("idea is to take results from file if "
                                      "simulations are too long. will be easy "
                                      "for tracewin.")
