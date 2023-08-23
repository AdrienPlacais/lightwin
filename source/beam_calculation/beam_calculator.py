#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:26:52 2023.

@author: placais

This module holds the base class `BeamCalculator`. It computes the propagation
of the beam in a `ListOfElements`, possibly with a specific
`SetOfCavitySettings` (optimisation process). It should create
`SimulationOutput`.

"""
import logging
from typing import Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
from functools import partial

from beam_calculation.output import SimulationOutput
from failures.set_of_cavity_settings import SetOfCavitySettings
from core.elements import _Element
from core.list_of_elements import ListOfElements, equiv_elt
from core.accelerator import Accelerator


@dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    out_folder: str

    def __post_init__(self):
        """List the mandatory arguments."""
        self.id: str = self.__repr__()

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

    @abstractmethod
    def _generate_simulation_output(self, *args: Any) -> SimulationOutput:
        """Transform the output of `BeamCalculator` to a `SimulationOutput`."""

    def _generate_element_to_index_func(self, elts: ListOfElements
                                        ) -> Callable[[_Element, str | None],
                                                      int | slice]:
        """Create the func to easily get data at proper mesh index."""
        shift = elts[0].beam_calc_param[self.id].s_in
        return partial(_element_to_index, _elts=elts, _shift=shift,
                       _solver_id=self.id)

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


def _element_to_index(_elts: ListOfElements, _shift: int, _solver_id: str,
                      elt: _Element | str, pos: str | None = None
                      ) -> int | slice:
    """
    Convert `elt` + `pos` into a mesh index.

    This way, you can call `.get('w_kin', elt='FM5', pos='out')` and
    systematically get the energy at the exit of FM5, whatever the
    `BeamCalculator` or the mesh size is.

    Parameters
    ----------
    _elts : ListOfElements
        List of `_Element`s where `elt` should be. Must be set by a
        `functools.partial`.
    _shift : int
        Mesh index of first `_Element`. Used when the first `_Element` of
        `_elts` is not the first of the `Accelerator`. Must be set by
        `functools.partial`.
    _solver_id : str
        Name of the solver, to identify and take the proper
        `SingleElementBeamParameters`.
    elt : _Element | str
        Element of which you want the index.
    pos : 'in' | 'out' | None, optional
        Index of entry or exit of the `_Element`. If None, return full
        indexes array. The default is None.

    """
    if isinstance(elt, str):
        elt = equiv_elt(elts=_elts, elt=elt)

    beam_calc_param = elt.beam_calc_param[_solver_id]
    if pos is None:
        return slice(beam_calc_param.s_in - _shift,
                     beam_calc_param.s_out - _shift + 1)
    elif pos == 'in':
        return beam_calc_param.s_in - _shift
    elif pos == 'out':
        return beam_calc_param.s_out - _shift
    else:
        logging.error(f"{pos = }, while it must be 'in', 'out' or None")
