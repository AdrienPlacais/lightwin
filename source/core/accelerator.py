#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021.

@author: placais

TODO : Check if _check_consistency_phases message still relatable
TODO : compute_transfer_matrices: simplify, add a calculation of missing phi_0
at the end
"""
import os.path
import logging
from typing import Any
import numpy as np


import config_manager as con
import tracewin.interface
import tracewin.load
from beam_calculation.output import SimulationOutput
import util.converters as convert
from util.helper import recursive_items, recursive_getter
from core import particle
from core.elements import _Element, FieldMapPath, Freq, Lattice
from core.list_of_elements import ListOfElements, elt_at_this_s_idx, \
    equiv_elt
from core.emittance import beam_parameters_all


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, dat_filepath: str, project_folder: str,
                 name: str) -> None:
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name

        # Prepare files and folders
        self.files = {
            'dat_filepath': os.path.abspath(dat_filepath),
            'orig_dat_folder': os.path.abspath(os.path.dirname(dat_filepath)),
            'project_folder': project_folder,
            'dat_filecontent': None,
            'field_map_folder': None,
            'out_lw': None,
            'out_tw': None}

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent = tracewin.load.dat_file(dat_filepath)
        elts = tracewin.interface.create_structure(dat_filecontent)
        elts = self._handle_paths_and_folders(elts)

        elts_copy = elts.copy()

        # elts, sections, lattices, freqs = _sections_lattices(elts)
        elts, elts_by_section_and_lattice, elts_by_lattice, freqs = \
            structured(elts_copy)
        set_structure_related_indexes(elts, elts_by_section_and_lattice,
                                      elts_by_lattice)

        self.elts = ListOfElements(elts, w_kin=con.E_MEV, phi_abs=0.,
                                   first_init=True)

        self.elements = {'l_lattices': elts_by_lattice,
                         'l_sections': elts_by_section_and_lattice}

        tracewin.interface.set_all_electric_field_maps(
            self.files, elts_by_section_and_lattice, freqs, con.F_BUNCH_MHZ)
        self.elts.set_absolute_positions()
        last_idx = self.elts.set_indexes()

        self.files['dat_filecontent'] = dat_filecontent

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., con.E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'tm_cumul': np.full((last_idx + 1, 2, 2), np.NaN),
            'tm_indiv': np.full((last_idx + 1, 2, 2), np.NaN),
        }
        self.transf_mat['tm_indiv'][0, :, :] = np.eye(2)

        # Beam parameters: emittance, Twiss
        self.beam_param = {
            "twiss": {"twiss_zdelta": np.full((last_idx + 1, 3), np.NaN),
                      "twiss_z": np.full((last_idx + 1, 3), np.NaN),
                      "twiss_w": np.full((last_idx + 1, 3), np.NaN)},
            "eps": {"eps_zdelta": np.full((last_idx + 1), np.NaN),
                    "eps_z": np.full((last_idx + 1), np.NaN),
                    "eps_w": np.full((last_idx + 1), np.NaN)},
            "envelopes": {
                "envelopes_zdelta": np.full((last_idx + 1, 2), np.NaN),
                "envelopes_z": np.full((last_idx + 1, 2), np.NaN),
                "envelopes_w": np.full((last_idx + 1, 2), np.NaN)},
            "mismatch_factor": np.full((last_idx + 1), np.NaN),
            "sigma_matrix": np.full((last_idx + 1), np.NaN),
        }
        # Define some shortcuts
        self._d_special_getters = self._create_special_getters()

        self.synch.init_abs_z(self.get('abs_mesh', remove_first=True))
        # Check that LW and TW computes the phases in the same way (abs or rel)
        self._check_consistency_phases()

        self._l_cav = self.elts.l_cav

    @property
    def l_cav(self):
        """Shortcut to easily get list of cavities."""
        return self.elts.l_cav

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: tuple[str, ...], to_numpy: bool = True, **kwargs: dict
            ) -> Any:
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if key in self._d_special_getters:
                val[key] = self._d_special_getters[key](self)
                continue

            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        **kwargs)

        # Convert to list, and to numpy array if necessary
        out = [val[key] for key in keys]
        if to_numpy:
            # VisibleDeprecationWarning with Python < 3.11?
            # Error with Python >= 3.11?
            # Try setting to_numpy to False.
            out = [np.array(val) if isinstance(val, list)
                   else val
                   for val in out]

        # Return as tuple or single value
        if len(keys) == 1:
            return out[0]
        # implicit else
        return tuple(out)

    # TODO add linac name in the subproject folder name
    def _handle_paths_and_folders(self, l_elts: list[_Element, ...]
                                  ) -> list[_Element]:
        """Make paths absolute, create results folders."""
        # First we take care of where results will be stored
        i = 0

        # i is useful if  you launch several simulations
        out_base = os.path.join(self.files['project_folder'], f"{i:06d}")

        while os.path.exists(out_base):
            i += 1
            out_base = os.path.join(self.files['project_folder'], f"{i:06d}")
        os.makedirs(out_base)
        self.files['out_lw'] = os.path.join(out_base, 'LW')
        self.files['out_tw'] = os.path.join(out_base, 'TW')

        # Now we handle where to look for the field maps
        field_map_basepaths = [basepath
                               for basepath in l_elts
                               if isinstance(basepath, FieldMapPath)]
        # FIELD_MAP_PATH are not physical elements, so we remove them
        for basepath in field_map_basepaths:
            l_elts.remove(basepath)

        # If no FIELD_MAP_PATH command was provided, we take field maps in the
        # .dat dir
        if len(field_map_basepaths) == 0:
            field_map_basepaths = [FieldMapPath(
                ['FIELD_MAP_PATH', self.files['orig_dat_folder']])]

        # If more than one field map folder is provided, raise an error
        msg = "Change of base folder for field maps currently not supported."
        assert len(field_map_basepaths) == 1, msg

        # Convert FieldMapPath objects into absolute paths
        field_map_basepaths = [
            os.path.join(self.files['orig_dat_folder'],
                         fm_path.path)
            for fm_path in field_map_basepaths]
        self.files['field_map_folder'] = field_map_basepaths[0]

        return l_elts

    def _create_special_getters(self) -> dict:
        """Create a dict of aliases that can be accessed w/ the get method."""
        _d_special_getters = {
            'alpha_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 0],
            'beta_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 1],
            'gamma_zdelta': lambda self:
                self.beam_param['twiss']['twiss_zdelta'][:, 2],
            'alpha_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 0],
            'beta_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 1],
            'gamma_z': lambda self:
                self.beam_param['twiss']['twiss_z'][:, 2],
            'alpha_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 0],
            'beta_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 1],
            'gamma_w': lambda self:
                self.beam_param['twiss']['twiss_w'][:, 2],
            'envelope_pos_zdelta': lambda self:
                self.beam_param['envelopes']['envelopes_zdelta'][:, 0],
            'envelope_energy_zdelta': lambda self:
                self.beam_param['envelopes']['envelopes_zdelta'][:, 1],
            'envelope_pos_z': lambda self:
                self.beam_param['envelopes']['envelopes_z'][:, 0],
            'envelope_energy_z': lambda self:
                self.beam_param['envelopes']['envelopes_z'][:, 1],
            'envelope_pos_w': lambda self:
                self.beam_param['envelopes']['envelopes_w'][:, 0],
            'envelope_energy_w': lambda self:
                self.beam_param['envelopes']['envelopes_w'][:, 1],
            'M_11': lambda self: self.transf_mat['tm_cumul'][:, 0, 0],
            'M_12': lambda self: self.transf_mat['tm_cumul'][:, 0, 1],
            'M_21': lambda self: self.transf_mat['tm_cumul'][:, 1, 0],
            'M_22': lambda self: self.transf_mat['tm_cumul'][:, 1, 1],
            'element number': lambda self: self.get('elt_idx') + 1,
        }
        return _d_special_getters

    def _check_consistency_phases(self) -> None:
        """Check that both TW and LW use absolute or relative phases."""
        flags_absolute = []
        for cav in self.l_cav:
            flags_absolute.append(cav.get('abs_phase_flag'))

        if con.FLAG_PHI_ABS and False in flags_absolute:
            logging.warning(
                "You asked LW a simulation in absolute phase, while there "
                + "is at least one cavity in relative phase in the .dat file "
                + "used by TW. Results won't match if there are faulty "
                + "cavities.")
        elif not con.FLAG_PHI_ABS and True in flags_absolute:
            logging.warning(
                "You asked LW a simulation in relative phase, while there "
                + "is at least one cavity in absolute phase in the .dat file "
                + "used by TW. Results won't match if there are faulty "
                + "cavities.")

    def keep_this(self, simulation_output: SimulationOutput,
                  l_elts: list[_Element]) -> None:
        """
        We save data into the appropriate objects.

        In particular:
            energy and phase into accelerator.synch
            rf field parameters, element transfer matrices into Elements
            global transfer matrices into Accelerator

        This function is called when the fitting is not required/is already
        finished.
        """
        idx_in = l_elts[0].idx['s_in']
        idx_out = l_elts[-1].idx['s_out'] + 1

        # Go across elements
        for elt, rf_field, cav_params in zip(l_elts,
                                             simulation_output.rf_fields,
                                             simulation_output.cav_params):
            idx_abs = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)
            idx_rel = range(elt.idx['s_in'] + 1 - idx_in,
                            elt.idx['s_out'] + 1 - idx_in)
            transf_mat_elt = simulation_output.tm_cumul[idx_rel]

            self.transf_mat['tm_indiv'][idx_abs] = transf_mat_elt
            elt.keep_rf_field(rf_field, cav_params)

        # Save into Accelerator
        self.transf_mat['tm_cumul'][idx_in:idx_out] \
            = simulation_output.tm_cumul
        self.beam_param["sigma_matrix"] = simulation_output.sigma_matrix
        # Mismatch will be None straight out of
        # ListOfElements._pack_into_single_dict method
        # We add it manually to results dict during the fitting process
        # (FaultScenario)
        mism = simulation_output.mismatch_factor
        if mism is not None:
            self.beam_param["mismatch_factor"] = mism

        # Save into Particle
        self.synch.keep_energy_and_phase(simulation_output,
                                         range(idx_in, idx_out))

        # Save into Accelerator
        gamma = convert.energy(simulation_output.get('w_kin'), "kin to gamma")
        d_beam_param = beam_parameters_all(simulation_output.eps_zdelta,
                                           simulation_output.twiss_zdelta,
                                           gamma)

        # Go across beam parameters (Twiss, emittance, long. envelopes)
        for item1 in self.beam_param.items():
            if not isinstance(item1[1], dict):
                continue
            # Go across phase spaces (z-z', z-delta, w-phi)
            for item2 in item1[1].items():
                item2[1][idx_in:idx_out] = d_beam_param[item1[0]][item2[0]]

    def elt_at_this_s_idx(self, s_idx: int, show_info: bool = False
                          ) -> _Element | None:
        """Give the element where the given index is."""
        return elt_at_this_s_idx(self.elts, s_idx, show_info)

    def equiv_elt(self, elt: _Element | str, to_index: bool = False
                  ) -> _Element | int | None:
        """Return an element from self.elts with the same name."""
        return equiv_elt(self.elts, elt, to_index)


def _sections_lattices(elts: list[_Element]
                       ) -> tuple[list[_Element], list[_Element],
                                  list[_Element], dict]:
    """Gather elements by section and lattice."""
    elts, structure = _prepare_sections_and_lattices(elts)

    lattices = []
    sections = []
    j = 0
    for i in range(structure['n_sections']):
        lattices = []
        n_lattice = structure['lattices'][i].n_lattice
        while j < structure['indices'][i]:
            lattices.append(elts[j:j + n_lattice])
            j += n_lattice
        sections.append(lattices)

    shift_lattice = 0
    j = 0
    for i, sec in enumerate(sections):
        for j, lattice in enumerate(sec):
            for elt in lattice:
                elt.idx['section'] = i
                elt.idx['lattice'] = j + shift_lattice
                elt.idx['elt_idx'] = elts.index(elt)
        shift_lattice += j + 1
    lattices = []
    for sec in sections:
        lattices += sec
    return elts, sections, lattices, structure['frequencies']


def _prepare_sections_and_lattices(elts: list[_Element]
                                   ) -> tuple[list[_Element], dict]:
    """
    Set sections, lattices and the different rf frequencies of the Accelerator.

    Parameters
    ----------
    elts : list[_Element]
        A list containing all the _Elements of the linac, including the 'fake'
        _Elements that are `Lattice` and `Freq` commands from the .dat file.
        TODO : create a elements._Command object?

    Returns
    -------
    elts : list[_Element]
        The same list, but with only real _Elements.
    structure : dict
        A dictionary containing the same _Elements, but grouped by Section
        and/or by Lattice. Also holds every Section rf frequency.

    """
    lattices = list(filter(lambda elt: isinstance(elt, Lattice), elts))
    frequencies = list(filter(lambda elt: isinstance(elt, Freq), elts))
    n_sections = len(frequencies)

    sections_changing_indexes = []
    number_of_sections_before_this_one = 0

    assert len(lattices) == len(frequencies)
    for latt, freq in zip(lattices, frequencies):
        idx_latt = elts.index(latt)
        idx_freq = elts.index(freq)
        if np.abs(idx_freq - idx_latt) != 1:
            logging.error("Commands LATTICE and FREQ do not follow each "
                          + "other. Check your .dat file.")

        sections_changing_indexes.append(
            idx_latt - 2 * number_of_sections_before_this_one)
        number_of_sections_before_this_one += 1

        elts.pop(idx_freq)
        elts.pop(idx_latt)
    sections_changing_indexes.pop(0)
    sections_changing_indexes.append(len(elts))

    return elts, {'lattices': lattices, 'frequencies': frequencies,
                  'indices': sections_changing_indexes,
                  'n_sections': n_sections}


def structured(elts: list[_Element]) -> tuple[list[_Element],
                                              list[list[_Element]],
                                              list[list[list[_Element]]],
                                              list[Freq]]:
    """
    Remove Freq/Lattice commands, regroup _Elements by section/lattice.

    Parameters
    ----------
    elts : list[_Element]
        Raw list of _Elements, as read from the .dat file.

    Returns
    -------
    elts : list[_Element]
        Same list as input, but without Freq and Lattice objects.
    elts_by_section_and_lattice : list[list[list[_Element]]]
        Level 1: Sections. Level 2: Lattices. Level 3: list of _Elements in the
        Lattice.
    elts_by_lattice : list[list[_Element]]
        Level 1: Lattices. Level 2: list of _Elements in the Lattice.
    frequencies: list[Freq]
        Contains the Frequency object corresponding to every Section.

    """
    lattices, frequencies = _lattices_and_frequencies(elts)

    _elts_by_section = _group_elements_by_section(elts, lattices)

    to_exclude = (Lattice, Freq)
    elts = _filter_out(elts, to_exclude)
    _elts_by_section = _filter_out(_elts_by_section, to_exclude)

    elts_by_section_and_lattice = _group_elements_by_section_and_lattice(
        _elts_by_section, lattices)
    elts_by_lattice = _group_elements_by_lattice(
        elts_by_section_and_lattice, lattices)

    return elts, elts_by_section_and_lattice, elts_by_lattice, frequencies


def set_structure_related_indexes(
        elts: list[_Element],
        elts_by_section_and_lattice: list[list[list[_Element]]],
        elts_by_lattice: list[list[_Element]]) -> None:
    """Set useful indexes, which are related to the structure of the linac."""
    _set_elt_index(elts)
    _set_section_index(elts_by_section_and_lattice)
    _set_lattice_index(elts_by_lattice)


def _set_elt_index(elts: list[_Element]) -> None:
    """Set `elt_indexes`, the absolute _Element's index."""
    for elt in elts:
        elt.idx['elt_idx'] = elts.index(elt)


def _set_section_index(elts_by_section_and_lattice: list[list[list[_Element]]]
                       ) -> None:
    """Set section indexes, which simply are the Section number of each elt."""
    for i, section in enumerate(elts_by_section_and_lattice):
        for lattice in section:
            for element in lattice:
                element.idx['section'] = i


def _set_lattice_index(elts_by_lattice: list[list[_Element]]
                       ) -> None:
    """
    Give each _Element its lattice number.

    This index is not reset to 0 at Sections changes.
    """
    for i, lattice in enumerate(elts_by_lattice):
        for elt in lattice:
            elt.idx['lattice'] = i


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
