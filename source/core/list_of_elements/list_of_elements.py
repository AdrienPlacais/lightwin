#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais

In this module we define the :class:`ListOfElements` object. It is a ``list``
of :class:`.Element`, with some additional methods.

Two objects can have a :class:`ListOfElements` as attribute:
    - :class:`.Accelerator`: holds all the :class:`.Element` of the linac.
    - :class:`.Fault`: it holds only a fraction of the linac
      :class:`.Element`. Beam will be propagated a huge number of times
      during optimisation process, so we recompute only the strict necessary.

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

from tracewin_utils.dat_files import (update_field_maps_in_dat,
                                      save_dat_filecontent_to_dat)
from tracewin_utils.interface import list_of_elements_to_command

from util.helper import recursive_items, recursive_getter


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, elts: list[Element],
                 input_particle: ParticleInitialState,
                 input_beam: BeamParameters, first_init: bool = True,
                 files: dict[str, str | list[list[str]]] | None = None
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
            List containing the element objects.
        input_particle : ParticleInitialState
            An object to hold initial energy and phase of the particle at the
            entry of the first element/
        input_beam : BeamParameters
            An object to hold emittances, Twiss, sigma beam matrix, etc at the
            entry of the first element.
        first_init : bool, optional
            To indicate if this a full linac or only a portion (fit process).
            The default is True.
        files : dict[str, str | list[list[str]]], optional
            A dictionary to hold information on the source and output
            files/folders of the object. The keys are:
                - ``dat_filepath``: path to the ``.dat`` file
                - ``elts_n_cmds``: list of objects representing dat content
                - ``out_folder``: where calculation results should be stored
                - ``dat_content``: list of list of str, holding content of the
                ``.dat``.

        """
        self.input_particle = input_particle
        self.input_beam = input_beam
        self.files = files

        super().__init__(elts)
        self.by_section_and_lattice: list[list[list[Element]]] | None = None
        self.by_lattice: list[list[Element]] | None = None

        if first_init:
            self._first_init()

        self._l_cav: list[FieldMap] = list(filter(
            lambda cav: isinstance(cav, FieldMap), self
            ))
        logging.info("Successfully created a ListOfElements with "
                     f"{self.w_kin_in = } MeV and {self.phi_abs_in = } rad.")

    @property
    def w_kin_in(self):
        """Get kinetic energy at entry of first element of self."""
        return self.input_particle.w_kin

    @property
    def phi_abs_in(self):
        """Get absolute phase at entry of first element of self."""
        return self.input_particle.phi_abs

    @property
    def tm_cumul_in(self):
        """Get transfer matrix at entry of first element of self."""
        return self.input_beam.zdelta.tm_cumul

    @property
    def l_cav(self):
        """Easy access to the list of cavities."""
        return self._l_cav

    @property
    def _stored_k_e(self) -> dict[FieldMap, float]:
        """Get the ``k_e`` properties from elements in self."""
        k_e = {cavity: cavity.get('k_e') for cavity in self.l_cav}
        return k_e

    @property
    def _stored_abs_phase_flag(self) -> dict[FieldMap, int]:
        """Get the ``abs_phase`` flags from elements in self."""
        abs_phase_flag = {cavity: int(config_manager.FLAG_PHI_ABS)
                          for cavity in self.l_cav}
        return abs_phase_flag

    @property
    def _stored_phi_0_abs(self):
        """Return the ``phi_0_abs`` properties from elements in self."""
        phi_0_abs = {cavity: cavity.get('phi_0_abs') for cavity in self.l_cav}
        return phi_0_abs

    @property
    def _stored_phi_0_abs_rephased(self):
        """
        Return the ``phi_0_abs`` from ``self``, rephased w.r.t ``phi_in``.

        Necessary for :class:`.TraceWin` and used in ``.dat`` files.
        Would mess with :class:`.Envelope1D`. Do not use it to update
        ``acc_field`` from :class:`.FieldMap`.

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

        This method also looks into the first :class:`.Element` of self. If the
        desired ``key`` is in this :class:`.Element`, we recursively get ``key``
        from every :class:`.Element` and concatenate the output.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        remove_first : bool, optional
            If you want to remove the first item of every :class:`.Element`
            ``key``.
            It the element is the first of the list, we do not remove its first
            item.  It is useful when the last item of an element is the same as
            the first item of the next element. For example, ``z_abs``. The
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
        self._set_element_indexes()

    def _set_element_indexes(self) -> None:
        """Set the element index."""
        elts_with_a_number = list(filter(
            lambda elt: elt.idx['increment_elt_idx'], self))

        for i, elt in enumerate(elts_with_a_number):
            elt.idx['elt_idx'] = i

    def store_settings_in_dat(self,
                              dat_filepath: str,
                              save: bool = True
                              ) -> None:
        """
        Update the dat file, save it if asked.

        Important notice
        ----------------
        The phases of the cavities are rephased if the first :class:`.Element`
        in self is not the first of the linac. This way, the beam enters each
        cavity with the intended phase in :class:`.TraceWin`.

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
