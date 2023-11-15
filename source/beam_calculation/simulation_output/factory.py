#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily generate the :class:`.SimulationOutput`.

This class should be subclassed by every :class:`.BeamCalculator` to match its
own specific outputs.

"""
from abc import ABC, abstractmethod, ABCMeta
from dataclasses import dataclass
from functools import partial
import logging
from typing import Callable

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.elements.element import Element
from core.list_of_elements.helper import equivalent_elt
from core.list_of_elements.list_of_elements import ListOfElements
from core.transfer_matrix.factory import TransferMatrixFactory
from core.beam_parameters.factory import BeamParametersFactory


@dataclass
class SimulationOutputFactory(ABC):
    """A base class for creation of :class:`.SimulationOutput`."""

    _is_3d: bool
    _is_multipart: bool
    _solver_id: str

    def __post_init__(self) -> None:
        """Create the factories.

        The created factories are :class:`.TransferMatrixFactory` and
        :class:`.BeamParametersFactory`. The sub-class that is used is declared
        in :meth:`._transfer_matrix_factory_class` and
        :meth:`._beam_parameters_factory_class`.

        """
        self.transfer_matrix_factory = self._transfer_matrix_factory_class(
            self._is_3d)
        self.beam_parameters_factory = self._beam_parameters_factory_class(
            self._is_3d,
            self._is_multipart)

    @property
    @abstractmethod
    def _transfer_matrix_factory_class(self) -> ABCMeta:
        """Declare the **class** of the transfer matrix factory."""

    @property
    @abstractmethod
    def _beam_parameters_factory_class(self) -> ABCMeta:
        """Declare the **class** of the beam parameters factory."""

    @abstractmethod
    def run(self, *args, **kwargs) -> SimulationOutput:
        """Create the :class:`.SimulationOutput`."""
        return SimulationOutput(*args, **kwargs)

    def _generate_element_to_index_func(self, elts: ListOfElements
                                        ) -> Callable[[Element, str | None],
                                                      int | slice]:
        """Create the func to easily get data at proper mesh index."""
        shift = elts[0].beam_calc_param[self._solver_id].s_in
        element_to_index = partial(_element_to_index,
                                   _elts=elts,
                                   _shift=shift,
                                   _solver_id=self._solver_id
                                   )
        return element_to_index


def _element_to_index(_elts: ListOfElements,
                      _shift: int,
                      _solver_id: str,
                      elt: Element | str,
                      pos: str | None = None,
                      return_elt_idx: bool = False,
                      ) -> int | slice:
    """
    Convert ``elt`` and ``pos`` into a mesh index.

    This way, you can call ``get('w_kin', elt='FM5', pos='out')`` and
    systematically get the energy at the exit of FM5, whatever the
    :class:`.BeamCalculator` or the mesh size is.

    .. todo::
        different functions, for different outputs. At least, an
        _element_to_index and a _element_to_indexes. And also a different
        function for when the index element is desired.

    Parameters
    ----------
    _elts : ListOfElements
        List of :class:`.Element` where ``elt`` should be. Must be set by a
        :func:`functools.partial`.
    _shift : int
        Mesh index of first :class:`.Element`. Used when the first
        :class:`.Element` of ``_elts`` is not the first of the
        :class:`.Accelerator`. Must be set by :func:`functools.partial`.
    _solver_id : str
        Name of the solver, to identify and take the proper
        :class:`.SingleElementBeamParameters`.
    elt : Element | str
        Element of which you want the index.
    pos : {'in', 'out} | None, optional
        Index of entry or exit of the :class:`.Element`. If None, return full
        indexes array. The default is None.
    return_elt_idx : bool, optional
        If True, the returned index is the position of the element in
        ``_elts``.

    Returns
    -------
    int | slice
        Index of range of indexes where ``elt`` is.

    """
    if isinstance(elt, str):
        elt = equivalent_elt(elts=_elts, elt=elt)

    beam_calc_param = elt.beam_calc_param[_solver_id]
    if return_elt_idx:
        return _elts.index(elt)

    if pos is None:
        return slice(beam_calc_param.s_in - _shift,
                     beam_calc_param.s_out - _shift + 1)
    if pos == 'in':
        return beam_calc_param.s_in - _shift
    if pos == 'out':
        return beam_calc_param.s_out - _shift

    logging.error(f"{pos = }, while it must be 'in', 'out' or None")
    raise IOError(f"{pos = }, while it must be 'in', 'out' or None")
