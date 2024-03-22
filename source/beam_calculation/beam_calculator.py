#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a base class for beam propagation computing tools.

Define the base class :class:`BeamCalculator`, which computes the propagation
of the beam in a :class:`.ListOfElements`, possibly with a specific
:class:`.SetOfCavitySettings` (optimisation process). It should return a
:class:`.SimulationOutput`.

.. todo::
    Precise that BeamParametersFactory and TransferMatrixFactory are mandatory.

"""
import datetime
import logging
import time
from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path

import config_manager as con
from beam_calculation.parameters.factory import \
    ElementBeamCalculatorParametersFactory
from beam_calculation.simulation_output.factory import SimulationOutputFactory
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.list_of_elements.factory import ListOfElementsFactory
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings


class BeamCalculator(ABC):
    """Store a beam dynamics solver and its results."""

    _ids = count(0)

    def __init__(self,
                 out_folder: Path | str,
                 default_field_map_folder: Path | str,
                 ) -> None:
        """Set ``id`` and factories.

        Parameters
        ----------
        out_folder : Path | str
            Name of the folder where results should be stored, for each
            :class:`.Accelerator` under study. This is a relative path.
        default_field_map_folder : Path | str
            Where to look for field map files by default.

        """
        self.id: str = f"{self.__class__.__name__}_{next(self._ids)}"

        if isinstance(out_folder, str):
            out_folder = Path(out_folder)
        self.out_folder = out_folder

        if isinstance(default_field_map_folder, str):
            default_field_map_folder = Path(default_field_map_folder)
        self.default_field_map_folder = \
            default_field_map_folder.resolve().absolute()

        self.simulation_output_factory: SimulationOutputFactory
        self.list_of_elements_factory: ListOfElementsFactory
        self.beam_calc_parameters_factory: \
            ElementBeamCalculatorParametersFactory
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
            default_field_map_folder=self.default_field_map_folder,
            load_field_maps=True,  # useless with TraceWin
            field_maps_in_3d=False,  # not implemented anyway
            # Different loading of field maps if Cython
            load_cython_field_maps=con.FLAG_CYTHON,
            elements_to_remove=(),
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
        """Init some :class:`BeamCalculator` solver parameters."""

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
                **kwargs: SimulationOutput | None,
                ) -> SimulationOutput:
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
            For calculation of miNonesmatch factors..

        Returns
        -------
        simulation_output : SimulationOutput
            Object holding simulation results.

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
        return simulation_output
