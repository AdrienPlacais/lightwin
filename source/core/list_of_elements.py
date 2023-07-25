#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais

In this module we define the ListOfElements objects. These objects are created
in two contexts:
    - Accelerator.elts: holds all the _Elements of the linac.
    - Fault.elts, also called in FaultScenario: it holds only a fraction of the
    linac _Elements. Beam will be propagated a huge number of time during
    optimisation process, so we recompute only the strict necessary.
"""
import logging
from typing import Any
from functools import partial
import numpy as np

import config_manager
from core.elements import _Element, Freq, Lattice
import tracewin.interface
from util.helper import recursive_items, recursive_getter


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, elts: list[_Element], w_kin: float, phi_abs: float,
                 tm_cumul: np.ndarray | None = None, first_init: bool = True,
                 ) -> None:
        """
        Create the object, encompassing all the linac or only a fraction.

        The first case is when you initialize an Accelerator and compute the
        baseline energy, phase, etc values.
        The second case is when you only recompute a fraction of the linac,
        which is part of the optimisation process.

        Parameters
        ----------
        elts : list[_Element]
            List containing the _Element objects.
        w_kin : float
            Kinetic energy at the entrance of the first _Element. Must be in
            MeV.
        phi_abs : float
            Absolute phase at the entrance of the first _Element. Must be in
            rad, and expressed relative to the bunch frequency.
        tm_cumul : np.ndarray, optional
            Cumulated transfer matrix (2, 2) at the entrance of the first
            _Element. The default is None.
        first_init : bool, optional
            To indicate if this a full linac or only a portion (fit process).
            The default is True.

        """
        super().__init__(elts)
        self.w_kin_in = w_kin
        self.phi_abs_in = phi_abs

        self.by_section_and_lattice: list[list[list[_Element]]] | None = None
        self.by_lattice: list[list[_Element]] | None = None
        self.tm_cumul_in: np.ndarray | None

        if first_init:
            logging.info("First initialisation of ListOfElements, ecompassing "
                         + "all linac. Also removing Lattice and Freq "
                         + "commands, setting Lattice/Section structures, "
                         + "_Elements names.")
            self._first_init(tm_cumul)

        else:
            logging.info(f"Initalisation of ListOfElements from already "
                         f"initialized elements: {elts[0]} to {elts[-1]}.")

            if np.any(np.isnan(tm_cumul)):
                logging.error("Previous transfer matrix was not calculated.")
            self.tm_cumul_in = tm_cumul

        logging.info(f"{w_kin = }, {phi_abs = }")
        self._l_cav = filter_cav(self)

    @property
    def l_cav(self):
        """Easy access to the list of cavities."""
        return self._l_cav

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or \
            key in recursive_items(vars(self[0]))

    def get(self, *keys: str, to_numpy: bool = True,
            remove_first: bool = False, **kwargs: bool | str | _Element | None
            ) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        This method also looks into the first `_Element` of `self`. If the
        desired `key` is in this `_Element`, we recursively get `key` from
        every `_Element` and concatenate the output.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        remove_first : bool, optional
            If you want to remove the first item of every `_Element`'s `key`.
            It the _Element is the first of the list, we do not remove its
            first item.
            It is useful when the last item of an `_Element` is the same as the
            first item of the next `_Element`. For example, `z_abs`. The
            default is False.
        **kwargs : bool | str | _Element | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            # Specific case: key is in _Element
            if self[0].has(key):
                for elt in self:
                    data = elt.get(key, to_numpy=False, **kwargs)

                    if remove_first and elt is not self[0]:
                        data = data[1:]
                    if isinstance(data, list):
                        val[key] += data
                        continue
                    val[key].append(data)
            else:
                val[key] = recursive_getter(key, vars(self), **kwargs)

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) for val in out]

        if len(keys) == 1:
            return out[0]
        return tuple(out)

    def _first_init(self, tm_cumul: np.ndarray | None) -> None:
        """Remove Lattice/Freq commands, set structure, some indexes."""
        by_section_and_lattice, by_lattice, freqs = self.set_structure()
        self.by_section_and_lattice = by_section_and_lattice
        self.by_lattice = by_lattice

        self.set_structure_related_indexes_of_elements()

        tracewin.interface.give_name(self)
        self._set_generic_electric_field_properties(
            freqs, freq_bunch=config_manager.F_BUNCH_MHZ)

        if tm_cumul is not None:
            logging.warning(
                "You do not need to provide a cumulated transfer matrix "
                + "to initialize this ListOfElements. It starts at the "
                + "beginning of the linac and the matrix should be the "
                + "eye matrix. Ignoring your input...")
        self.tm_cumul_in = np.eye(2)

    def set_structure(self) -> tuple[list[list[_Element]],
                                     list[list[list[_Element]]],
                                     list[Freq]]:
        """
        Use Freq/Lattice commands to set structure of the accelerator.

        Also remove these commands from the ListOfElements.

        Returns
        -------
        elts_by_section_and_lattice : list[list[list[_Element]]]
            Level 1: Sections. Level 2: Lattices. Level 3: list of _Elements
            in the Lattice.
        elts_by_lattice : list[list[_Element]]
            Level 1: Lattices. Level 2: list of _Elements in the Lattice.
        frequencies: list[Freq]
            Contains the Frequency object corresponding to every Section.

        """
        lattices, frequencies = _lattices_and_frequencies(self)
        if len(lattices) == 0:
            logging.error("No Lattice or Frequency object detected. Maybe the "
                          + "ListOfElements.structured method was already "
                          + "performed?")

        _elts_by_section = _group_elements_by_section(self, lattices)

        to_exclude = (Lattice, Freq)
        filtered_elts = _filter_out(self, to_exclude)
        _elts_by_section = _filter_out(_elts_by_section, to_exclude)
        super().__init__(filtered_elts)

        elts_by_section_and_lattice = _group_elements_by_section_and_lattice(
            _elts_by_section, lattices)
        elts_by_lattice = _group_elements_by_lattice(
            elts_by_section_and_lattice, lattices)

        return elts_by_section_and_lattice, elts_by_lattice, frequencies

    def set_structure_related_indexes_of_elements(self) -> None:
        """Set useful indexes, related to the structure of the linac."""
        # Absolute _Element index
        for elt in self:
            elt.idx['elt_idx'] = self.index(elt)

        # Number of the Section
        for i, section in enumerate(self.by_section_and_lattice):
            for lattice in section:
                for element in lattice:
                    element.idx['section'] = i

        # Number of the lattice (not reset to 0 @ each new lattice)
        for i, lattice in enumerate(self.by_lattice):
            for elt in lattice:
                elt.idx['lattice'] = i

    def _set_generic_electric_field_properties(self, freqs: list[Freq],
                                               freq_bunch: float) -> None:
        """Set omega0, n_cells in cavities (used with every BeamCalculator)."""
        assert len(self.by_section_and_lattice) == len(freqs)

        for i, section in enumerate(self.by_section_and_lattice):
            f_mhz = freqs[i].f_rf_mhz
            n_cells = int(f_mhz / freq_bunch)

            for lattice in section:
                for elt in lattice:
                    elt.acc_field.set_pulsation_ncell(f_mhz, n_cells)


def indiv_to_cumul_transf_mat(tm_cumul_in: np.ndarray,
                              r_zz_elt: list[np.ndarray], n_steps: int
                              ) -> np.ndarray:
    """
    Compute cumulated transfer matrix.

    Parameters
    ----------
    tm_cumul_in : np.ndarray
        Cumulated transfer matrix @ first element. Should be eye matrix if we
        are at the first element.
    r_zz_elt : list[np.ndarray]
        List of individual transfer matrix of the elements.
    n_steps : int
        Number of elements or elements slices.

    Returns
    -------
    cumulated_transfer_matrices : np.ndarray
        Cumulated transfer matrices.

    """
    cumulated_transfer_matrices = np.full((n_steps, 2, 2), np.NaN)
    cumulated_transfer_matrices[0] = tm_cumul_in
    for i in range(1, n_steps):
        cumulated_transfer_matrices[i] = \
            r_zz_elt[i - 1] @ cumulated_transfer_matrices[i - 1]
    return cumulated_transfer_matrices


def equiv_elt(elts: ListOfElements | list[_Element], elt: _Element | str,
              to_index: bool = False) -> _Element | int | None:
    """
    Return an element from elts that has the same name as elt.

    Important: this routine uses the name of the element and not its adress. So
    it will not complain if the _Element object that you asked for is not in
    this list of elements.
    In the contrary, it was meant to find equivalent cavities between different
    lists of elements.

    Parameters
    ----------
    elts : ListOfElements | list[_Element]
        List of elements where you want the position.
    elt : _Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an _Element. If it is an _Element, we take its name in the
        routine.
    to_index : bool, optional
        If True, the function returns the index of the _Element instead of the
        _Element itself.

    Returns
    -------
    _Element | int | None
        Equivalent _Element, position in list of elements, or None if not
        found.

    """
    if not isinstance(elt, str):
        elt = elt.get("elt_name")

    names = [x.get("elt_name") for x in elts]
    if elt not in names:
        logging.error(f"Element {elt} not found in this list of elements.")
        logging.debug(f"List of elements is:\n{elts}")
        return None

    idx = names.index(elt)
    if to_index:
        return idx
    return elts[idx]


def elt_at_this_s_idx(elts: ListOfElements | list[_Element, ...],
                      s_idx: int, show_info: bool = False
                      ) -> _Element | None:
    """Give the element where the given index is."""
    for elt in elts:
        if s_idx in range(elt.idx['s_in'], elt.idx['s_out']):
            if show_info:
                logging.info(
                    f"Mesh index {s_idx} is in {elt.get('elt_info')}.\n"
                    + f"Indexes of this elt: {elt.get('idx')}.")
            return elt

    logging.warning(f"Mesh index {s_idx} not found.")
    return None


def _lattices_and_frequencies(elts: list[_Element]
                              ) -> tuple[list[Lattice], list[Freq]]:
    """Get Lattice and Freq objects, which convey every Section information."""
    lattices = list(filter(lambda elt: isinstance(elt, Lattice), elts))
    frequencies = list(filter(lambda elt: isinstance(elt, Freq), elts))

    idx_lattice_change = [elts.index(latt) for latt in lattices]
    idx_freq_change = [elts.index(freq) for freq in frequencies]
    distance = np.array(idx_lattice_change) - np.array(idx_freq_change)
    if not np.all(distance == -1):
        logging.error("FREQ commands do no directly follow LATTICE commands. "
                      + "Check your .dat file.")
    return lattices, frequencies


def _group_elements_by_section(elts: list[_Element], lattices: list[Lattice]
                               ) -> list[list[_Element]]:
    """Regroup the _Element belonging to the same Section."""
    idx_lattice_change = [elts.index(latt) for latt in lattices]
    slice_starts = idx_lattice_change
    slice_ends = idx_lattice_change[1:] + [None]
    elts_grouped_by_section = [
        elts[start:stop]
        for start, stop in zip(slice_starts, slice_ends)]

    return elts_grouped_by_section


def _group_elements_by_lattice(
    elts_by_sec_and_latt: list[list[list[_Element]]], lattices: list[Lattice]
) -> list[list[_Element]]:
    """Regroup the _Element belonging to the same Lattice."""
    elts_grouped_by_lattice = []
    for by_section in elts_by_sec_and_latt:
        for by_lattice in by_section:
            elts_grouped_by_lattice.append(by_lattice)
    return elts_grouped_by_lattice


def _group_elements_by_section_and_lattice(
        elts_by_sec: list[list[_Element]], lattices: list[Lattice]
        ) -> list[list[list[_Element]]]:
    """Regroup _Elements by Section and then by Lattice."""
    elts_by_section_and_lattice = [
        _slice(elts_of_a_section, n_in_slice=latt.n_lattice)
        for elts_of_a_section, latt in zip(elts_by_sec, lattices)]
    return elts_by_section_and_lattice


def _slice(unsliced: list, n_in_slice: int) -> list[list]:
    """Convert a list to a list of sublist of length n_in_slice."""
    if len(unsliced) % n_in_slice != 0:
        logging.error("Number of items per slice is not a multiple of the "
                      + "total length of the original list.")
    n_slices = len(unsliced) // n_in_slice
    sliced = [unsliced[i * n_in_slice:(i + 1) * n_in_slice]
              for i in range(n_slices)]
    return sliced


# actually, type of elements and outputs is Nested[list[_Element]]
def _filter_out(elements: Any, to_exclude: tuple[type]) -> Any:
    """Filter out `to_exclude` types while keeping the input list structure."""
    if isinstance(elements[0], list):
        return [_filter_out(sub, to_exclude) for sub in elements]

    elif isinstance(elements, list):
        return list(filter(lambda elt: not isinstance(elt, to_exclude),
                           elements))
    else:
        raise TypeError("Wrong type for data filtering.")

    return elements


def filter_elts(elts: ListOfElements | list[_Element], key: str, val: Any
                ) -> list[_Element]:
    """Shortcut for filtering elements according to (key, val)."""
    return list(filter(lambda elt: elt.get(key) == val, elts))


filter_cav = partial(filter_elts, key='nature', val='FIELD_MAP')
