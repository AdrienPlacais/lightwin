#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais

In this module we define the `ListOfElements` object. This type of object is
a list of `Element`s, with some additional methods.

It is created in two contexts:
    - `Accelerator.elts`: holds all the `Element`s of the linac.
    - `Fault.elts`, also called in `FaultScenario`: it holds only a fraction of
    the linac `Element`s. Beam will be propagated a huge number of time during
    optimisation process, so we recompute only the strict necessary.

.. todo::
    Delete ``dat_content``, which does the same thing as ``elts_n_cmds`` but
    less good

"""
import logging
from typing import Any
from functools import partial
import numpy as np

import config_manager

from core.elements.element import Element
from core.elements.field_map import FieldMap

from core.beam_parameters import BeamParameters
from core.particle import ParticleInitialState
from core.electric_field import phi_0_abs_with_new_phase_reference

from tracewin_utils.dat_files import (give_name,
                                      update_field_maps_in_dat,
                                      save_dat_filecontent_to_dat)
from tracewin_utils.interface import list_of_elements_to_command

from util.helper import recursive_items, recursive_getter


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, elts: list[Element],
                 input_particle: ParticleInitialState,
                 input_beam: BeamParameters, first_init: bool = True,
                 files: dict[str, str | list[list[str]]] = None
                 ) -> None:
        """
        Create the object, encompassing all the linac or only a fraction.

        The first case is when you initialize an Accelerator and compute the
        baseline energy, phase, etc values.
        The second case is when you only recompute a fraction of the linac,
        which is part of the optimisation process.

        Parameters
        ----------
        elts : list[Element]
            List containing the Element objects.
        input_particle : ParticleInitialState
            An object to hold initial energy and phase of the particle at the
            entry of the first `Element`.
        input_beam : BeamParameters
            An object to hold emittances, Twiss, sigma beam matrix, etc at the
            entry of the first `Element`.
        first_init : bool, optional
            To indicate if this a full linac or only a portion (fit process).
            The default is True.
        files : dict[str, str | list[list[str]]], optional
            A dictionary to hold information on the source and output
            files/folders of the object. The keys are:
                dat_filepath : path to the `.dat` file
                dat_content : list of list of str, holding content of the `dat`
                elts_n_cmds : list of objects representing dat content
                out_folder : where calculation results should be stored

        """
        self.input_particle = input_particle
        self.input_beam = input_beam
        self.files = files

        super().__init__(elts)
        self.by_section_and_lattice: list[list[list[Element]]] | None = None
        self.by_lattice: list[list[Element]] | None = None

        if first_init:
            self._first_init()

        self._l_cav = filter_cav(self)
        logging.info("Successfully created a `ListOfElements` with "
                     f"{self.w_kin_in = } MeV and {self.phi_abs_in = } rad.")

    @property
    def w_kin_in(self):
        return self.input_particle.w_kin

    @property
    def phi_abs_in(self):
        return self.input_particle.phi_abs

    @property
    def tm_cumul_in(self):
        return self.input_beam.zdelta.tm_cumul

    @property
    def l_cav(self):
        """Easy access to the list of cavities."""
        return self._l_cav

    @property
    def _stored_k_e(self) -> dict[FieldMap, float]:
        """Get the `k_e` properties from `Element`s of `self`."""
        k_e = {cavity: cavity.get('k_e') for cavity in self.l_cav}
        return k_e

    @property
    def _stored_abs_phase_flag(self) -> dict[FieldMap, float]:
        """Get the `abs_phase` flags from `Element`s of `self`."""
        abs_phase_flag = {cavity: int(config_manager.FLAG_PHI_ABS)
                          for cavity in self.l_cav}
        return abs_phase_flag

    @property
    def _stored_phi_0_abs(self):
        """Return the `phi_0_abs` properties from `self`."""
        phi_0_abs = {cavity: cavity.get('phi_0_abs') for cavity in self.l_cav}
        return phi_0_abs

    @property
    def _stored_phi_0_abs_rephased(self):
        """
        Return the `phi_0_abs` properties from `self`, rephased w.r.t `phi_in`.

        Necessary for `TraceWin` and used in `.dat` files.
        Would mess with `Envelope1D`, do not use it to update `acc_field` from
        `FieldMap`.

        """
        delta_phi_bunch = self.input_particle.phi_abs
        phi_0_abs_rephased = {
            cavity: phi_0_abs_with_new_phase_reference(
                phi_0_abs,
                delta_phi_bunch * cavity.acc_field.n_cell
            )
            for cavity, phi_0_abs in self._stored_phi_0_abs.items()
        }
        return phi_0_abs_rephased

    @property
    def _stored_phi_0_rel(self):
        """Return the `phi_0_rel` properties from `self`."""
        phi_0_abs = {cavity: cavity.get('phi_0_rel') for cavity in self.l_cav}
        return phi_0_abs

    @property
    def tracewin_command(self) -> list[str]:
        """Create the command to give proper initial parameters to TraceWin."""
        dat_filepath = self.get('dat_filepath', to_numpy=False)
        _tracewin_command = [
            command_bit
            for command in [list_of_elements_to_command(dat_filepath),
                            self.input_particle.tracewin_command,
                            self.input_beam.tracewin_command]
            for command_bit in command]

        return _tracewin_command

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or \
            key in recursive_items(vars(self[0]))

    def get(self, *keys: str, to_numpy: bool = True,
            remove_first: bool = False, **kwargs: bool | str | Element | None
            ) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        This method also looks into the first `Element` of `self`. If the
        desired `key` is in this `Element`, we recursively get `key` from
        every `Element` and concatenate the output.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        remove_first : bool, optional
            If you want to remove the first item of every `Element`'s `key`.
            It the Element is the first of the list, we do not remove its
            first item.
            It is useful when the last item of an `Element` is the same as the
            first item of the next `Element`. For example, `z_abs`. The
            default is False.
        **kwargs : bool | str | Element | None
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

            # Specific case: key is in Element
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

    def _first_init(self) -> None:
        """Set structure, elements name, some indexes."""
        by_section = _group_elements_by_section(self)
        self.by_lattice = _group_elements_by_lattice(self)
        self.by_section_and_lattice = _group_elements_by_section_and_lattice(
            by_section)

        elts_with_a_number = list(filter(
            lambda elt: elt.idx['increment_elt_idx'],
            self))
        for i, elt in enumerate(elts_with_a_number):
            elt.idx['elt_idx'] = i
        give_name(self)

    def store_settings_in_dat(self,
                              dat_filepath: str,
                              save: bool = True
                              ) -> None:
        """
        Update the dat file, save it if asked.

        Important notice
        ----------------
        The phases of the cavities are rephased if the first `Element` in
        `self` is not the first of the linac. This way, the beam enters each
        cavity with the intended phase in `TraceWin`.

        """
        new_phases = self._stored_phi_0_rel
        if config_manager.FLAG_PHI_ABS:
            new_phases = self._stored_phi_0_abs_rephased
        update_field_maps_in_dat(
            self,
            new_phases=new_phases,
            new_k_e=self._stored_k_e,
            new_abs_phase_flag=self._stored_abs_phase_flag)
        if not save:
            return

        self.files['dat_filepath'] = dat_filepath
        dat_content = [elt_or_cmd.line
                       for elt_or_cmd in self.files['elts_n_cmds']]
        save_dat_filecontent_to_dat(dat_content, dat_filepath)


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


def equiv_elt(elts: ListOfElements | list[Element], elt: Element | str,
              to_index: bool = False) -> Element | int | None:
    """
    Return an element from elts that has the same name as elt.

    Important: this routine uses the name of the element and not its adress. So
    it will not complain if the Element object that you asked for is not in
    this list of elements.
    In the contrary, it was also meant to find equivalent cavities between
    different lists of elements.

    Parameters
    ----------
    elts : ListOfElements | list[Element]
        List of elements where you want the position.
    elt : Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an Element. If it is an Element, we take its name in the
        routine. Magic keywords 'first', 'last' are also accepted.
    to_index : bool, optional
        If True, the function returns the index of the Element instead of the
        Element itself.

    Returns
    -------
    Element | int | None
        Equivalent Element, position in list of elements, or None if not
        found.

    """
    if not isinstance(elt, str):
        elt = elt.get("elt_name")

    magic_keywords = {"first": 0, "last": -1}
    names = [x.get("elt_name") for x in elts]

    if elt in names:
        idx = names.index(elt)
    elif elt in magic_keywords:
        idx = magic_keywords[elt]
    else:
        logging.error(f"Element {elt} not found in this list of elements.")
        logging.debug(f"List of elements is:\n{elts}")
        return None

    if to_index:
        return idx
    return elts[idx]


def elt_at_this_s_idx(elts: ListOfElements | list[Element, ...],
                      s_idx: int, show_info: bool = False
                      ) -> Element | None:
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


def _group_elements_by_section(elts: list[Element],
                               ) -> list[list[Element]]:
    """Group elements by section."""
    n_sections = elts[-1].idx['section'] + 1
    by_section = [
        list(filter(lambda elt: elt.idx['section'] == current_section, elts))
        for current_section in range(n_sections)
    ]
    return by_section


def _group_elements_by_section_and_lattice(
        by_section: list[list[Element]],
) -> list[list[list[Element]]]:
    """Regroup Elements by Section and then by Lattice."""
    by_section_and_lattice = [
        _group_elements_by_lattice(section)
        for section in by_section
    ]
    return by_section_and_lattice


def _group_elements_by_lattice(elts: list[Element],
                               ) -> list[list[Element]]:
    """Regroup the Element belonging to the same Lattice."""
    idx_first_lattice = elts[0].idx['lattice']
    n_lattices = elts[-1].idx['lattice'] + 1
    by_lattice = [
        list(filter(lambda elt: (elt.idx['lattice'] is not None
                                 and elt.idx['lattice'] == current_lattice),
                    elts))
        for current_lattice in range(idx_first_lattice, n_lattices)
    ]
    return by_lattice


def _slice(unsliced: list, n_in_slice: int) -> list[list]:
    """Convert a list to a list of sublist of length n_in_slice."""
    if len(unsliced) % n_in_slice != 0:
        logging.error("Number of items per slice is not a multiple of the "
                      + "total length of the original list.")
    n_slices = len(unsliced) // n_in_slice
    sliced = [unsliced[i * n_in_slice:(i + 1) * n_in_slice]
              for i in range(n_slices)]
    return sliced


# actually, type of elements and outputs is Nested[list[Element]]
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


def filter_elts(elts: ListOfElements | list[Element], key: str, val: Any
                ) -> list[Element]:
    """Shortcut for filtering elements according to (key, val)."""
    return list(filter(lambda elt: elt.get(key) == val, elts))


filter_cav = partial(filter_elts, key='nature', val='FIELD_MAP')
