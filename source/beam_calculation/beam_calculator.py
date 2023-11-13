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

from beam_calculation.output import SimulationOutput

from failures.set_of_cavity_settings import SetOfCavitySettings

from core.elements.element import Element
from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import equivalent_elt
from core.accelerator import Accelerator
from core.beam_parameters.factory import (
    BeamParametersFactory,
    InitialBeamParametersFactory,
)
from core.transfer_matrix.factory import TransferMatrixFactory


@dataclass
class BeamCalculator(ABC):
    """A generic class to store a beam dynamics solver and its results."""

    out_folder: str

    def __post_init__(self):
        """Set ``id`` and :class:`.InitialBeamParametersFactory`."""
        self.id: str = self.__repr__()
        self.initial_beam_parameters_factory = InitialBeamParametersFactory(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation
        )
        self.beam_parameters_factory: BeamParametersFactory
        self.transfer_matrix_factory: TransferMatrixFactory

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
                                        ) -> Callable[[Element, str | None],
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


def _element_to_index(_elts: ListOfElements,
                      _shift: int,
                      _solver_id: str,
                      elt: Element | str,
                      pos: str | None = None,
                      return_elt_idx: bool = False,
                      ) -> int | slice:
    """
    Convert ``elt`` and ``pos`` into a mesh index.

    This way, you can call :func:`get('w_kin', elt='FM5', pos='out')` and
    systematically get the energy at the exit of FM5, whatever the
    :class:`BeamCalculator` or the mesh size is.

    Parameters
    ----------
    _elts : ListOfElements
        List of :class:`Element` where ``elt`` should be. Must be set by a
        :func:`functools.partial`.
    _shift : int
        Mesh index of first :class:`Element`. Used when the first
        :class:`Element` of ``_elts`` is not the first of the
        :class:`Accelerator`. Must be set by :func:`functools.partial`.
    _solver_id : str
        Name of the solver, to identify and take the proper
        :class:`SingleElementBeamParameters`.
    elt : Element | str
        Element of which you want the index.
    pos : 'in' | 'out' | None, optional
        Index of entry or exit of the :class:`Element`. If None, return full
        indexes array. The default is None.
    return_elt_idx : bool, optional
        If True, the returned index is the position of the element in
        ``_elts``.

    """
    if isinstance(elt, str):
        elt = equivalent_elt(elts=_elts, elt=elt)

    beam_calc_param = elt.beam_calc_param[_solver_id]
    if return_elt_idx:
        return _elts.index(elt)
    if pos is None:
        return slice(beam_calc_param.s_in - _shift,
                     beam_calc_param.s_out - _shift + 1)
    elif pos == 'in':
        return beam_calc_param.s_in - _shift
    elif pos == 'out':
        return beam_calc_param.s_out - _shift
    else:
        logging.error(f"{pos = }, while it must be 'in', 'out' or None")
        raise IOError(f"{pos = }, while it must be 'in', 'out' or None")
