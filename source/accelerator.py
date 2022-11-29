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
import numpy as np
import tracewin_interface as tw
import particle
from constants import E_MEV, FLAG_PHI_ABS
import elements
from list_of_elements import ListOfElements
from emittance import beam_parameters_all, mismatch_factor
from helper import kin_to_gamma, printc, recursive_items, recursive_getter


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, dat_filepath, name):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent, l_elts = tw.load_dat_file(dat_filepath)
        l_elts, l_secs, l_latts, freqs = _sections_lattices(l_elts)

        # Create a the list containing all the elements
        self.elts = ListOfElements(l_elts, w_kin=E_MEV, phi_abs=0., idx_in=0)

        self.elements = {'l_lattices': l_latts, 'l_sections': l_secs}

        tw.load_filemaps(dat_filepath, dat_filecontent,
                         self.elements['l_sections'], freqs)
        tw.give_name(l_elts)

        self.files = {
            'project_folder': os.path.dirname(dat_filepath),
            'dat_filecontent': dat_filecontent,
            'results_folder': os.path.dirname(dat_filepath) + '/results_lw/'}

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'tm_cumul': np.full((last_idx + 1, 2, 2), np.NaN),
            'tm_indiv': np.full((last_idx + 1, 2, 2), np.NaN),
            'first_calc?': True,
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
            "mismatch factor": np.full((last_idx + 1), np.NaN)
        }
        printc('Warning! accelerator. keys of beam param not gettable.')

        self.synch.init_abs_z(self.get('abs_mesh', remove_first=True))
        # Check that LW and TW computes the phases in the same way (abs or rel)
        self._check_consistency_phases()

    def has(self, key):
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys, to_numpy=True, **kwargs):
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue
            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        **kwargs)

        # Convert to list, and to numpy array if necessary
        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list)
                   else val
                   for val in out]

        # Return as tuple or single value
        if len(keys) == 1:
            return out[0]
        # implicit else
        return tuple(out)

    def _set_indexes_and_abs_positions(self):
        """Init solvers, set indexes and absolute positions of elements."""
        pos = {'in': 0., 'out': 0.}
        idx = {'in': 0, 'out': 0}

        for elt in self.elts:
            elt.init_solvers()

            pos['in'] = pos['out']
            pos['out'] += elt.length_m
            elt.solver_param['abs_mesh'] = elt.get('rel_mesh') + pos['in']

            idx['in'] = idx['out']
            idx['out'] += elt.get('n_steps')
            elt.idx['s_in'], elt.idx['s_out'] = idx['in'], idx['out']
        return idx['out']

    def _check_consistency_phases(self):
        """Check that both TW and LW use absolute or relative phases."""
        cavities = self.elements_of(nature='FIELD_MAP')
        flags_absolute = []
        for cav in cavities:
            flags_absolute.append(cav.get('abs_phase_flag'))

        if FLAG_PHI_ABS and False in flags_absolute:
            printc("Accelerator._check_consistency_phases warning: ",
                   opt_message="you asked LW a simulation in absolute phase,"
                   + " while there is at least one cavity in relative phase in"
                   + " the .dat file used by TW. Results won't match if there"
                   + " are faulty cavities.\n")
        elif not FLAG_PHI_ABS and True in flags_absolute:
            printc("Accelerator._check_consistency_phases warning: ",
                   opt_message="you asked LW a simulation in relative phase,"
                   + " while there is at least one cavity in absolute phase in"
                   + " the .dat file used by TW. Results won't match if there"
                   + " are faulty cavities.\n")

    def save_results(self, results, l_elts):
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
        for elt, rf_field, cav_params in zip(l_elts, results['rf_fields'],
                                             results["cav_params"]):
            idx_abs = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)
            idx_rel = range(elt.idx['s_in'] + 1 - idx_in,
                            elt.idx['s_out'] + 1 - idx_in)
            transf_mat_elt = results["r_zz_cumul"][idx_rel]

            self.transf_mat['tm_indiv'][idx_abs] = transf_mat_elt
            elt.keep_rf_field(rf_field, cav_params)

        # Save into Accelerator
        self.transf_mat['tm_cumul'][idx_in:idx_out] = results["r_zz_cumul"]

        # Save into Particle
        self.synch.keep_energy_and_phase(results, range(idx_in, idx_out))

        # Save into Accelerator
        gamma = kin_to_gamma(np.array(results["w_kin"]))
        d_beam_param = beam_parameters_all(results["d_zdelta"], gamma)

        # Go across beam parameters (Twiss, emittance, long. envelopes)
        for item1 in self.beam_param.items():
            if not isinstance(item1[1], dict):
                continue
            # Go across phase spaces (z-z', z-delta, w-phi)
            for item2 in item1[1].items():
                item2[1][idx_in:idx_out] = d_beam_param[item1[0]][item2[0]]

    def get_from_elements(self, attribute, key=None, other_key=None):
        """
        Return attribute from all elements in list_of_elements.

        Parameters
        ----------
        attribute : string
            Name of the desired attribute.
        key : string, optional
            If attribute is a dict, key must be provided. Default is None.
        other_key : string, optional
            If attribute[key] is a dict, a second key must be provided. Default
            is None.
        """
        list_of_keys = vars(self.elts[0])
        data_nature = str(type(list_of_keys[attribute]))

        dict_data_getter = {
            "<class 'dict'>": lambda elt_dict, key: elt_dict[key],
            "<class 'electric_field.RfField'>": lambda elt_class, key:
                vars(elt_class)[key],
            "<class 'numpy.ndarray'>": lambda data_new, key: data_new,
            "<class 'float'>": lambda data_new, key: data_new,
            "<class 'str'>": lambda data_new, key: data_new,
        }
        fun = dict_data_getter[data_nature]

        data_out = []
        for elt in self.elts:
            list_of_keys = vars(elt)
            piece_of_data = fun(list_of_keys[attribute], key)
            if isinstance(piece_of_data, dict):
                piece_of_data = piece_of_data[other_key]
            data_out.append(piece_of_data)

        # Concatenate into a single matrix
        if attribute == 'transfer_matrix':
            data_array = data_out[0]
            for i in range(1, len(data_out)):
                data_array = np.vstack((data_array, data_out[i]))
        # Transform into array
        else:
            data_array = np.array(data_out)
        return data_array

    def elements_of(self, nature, sub_list=None):
        """
        Return a list of elements of nature 'nature'.

        Parameters
        ----------
        nature : string
            Nature of the elements you want, eg FIELD_MAP or DRIFT.
        sub_list : list, optional
            List of elements (eg module) if you want the elements only in this
            module.

        Returns
        -------
        list_of : list of Element
            List of all the Elements which have a nature 'nature'.
        """
        if sub_list is None:
            sub_list = self.elts
        list_of = list(filter(lambda elt: elt.get('nature') == nature,
                              sub_list))
        return list_of

    def where_is_this_index(self, idx, showinfo=False):
        """Give the element where the given index is."""
        found, elt = False, None

        for elt in self.elts:
            if idx in range(elt.idx['s_in'], elt.idx['s_out']):
                found = True
                break

        if showinfo:
            if found:
                print(f"Mesh index {idx} is in {elt.get('elt_info')}.")
                print(f"Indexes of this elt: {elt.get('idx')}.\n")
            else:
                print(f"Mesh index {idx} does not belong to any element.")
        return elt if found else None

    def compute_mismatch(self, ref_linac):
        """Compute mismatch factor between this non-nominal linac and a ref."""
        assert self.name != 'Working'
        assert ref_linac.name == 'Working'
        self.beam_param["mismatch factor"] = \
                mismatch_factor(ref_linac.beam_param["twiss"]["twiss_z"],
                                self.beam_param["twiss"]["twiss_z"], transp=True)


def _sections_lattices(l_elts):
    """Gather elements by section and lattice."""
    l_elts, dict_struct = _prepare_sections_and_lattices(l_elts)

    lattices = []
    sections = []
    j = 0
    for i in range(dict_struct['n_sections']):
        lattices = []
        n_lattice = dict_struct['lattices'][i].n_lattice
        while j < dict_struct['indices'][i]:
            lattices.append(l_elts[j:j + n_lattice])
            j += n_lattice
        sections.append(lattices)

    shift_lattice = 0
    for i, sec in enumerate(sections):
        for j, lattice in enumerate(sec):
            for elt in lattice:
                elt.idx['section'] = i
                elt.idx['lattice'] = j
                elt.idx['elt_idx'] = l_elts.index(elt)
        shift_lattice += j + 1
    lattices = []
    for sec in sections:
        lattices += sec
    return l_elts, sections, lattices, dict_struct['frequencies']


def _prepare_sections_and_lattices(l_elts):
    """
    Save info on the accelerator structure.

    In particular: in every section, the number of elements per lattice,
    the rf frequency of cavities, the position of the delimitations between
    two sections.
    """
    lattices = [
        lattice
        for lattice in l_elts
        if isinstance(lattice, elements.Lattice)
    ]
    frequencies = [
        frequency
        for frequency in l_elts
        if isinstance(frequency, elements.Freq)
    ]
    n_sections = len(lattices)
    assert n_sections == len(frequencies)

    idx_of_section_changes = []
    n_secs_before_current = 0
    for i in range(n_sections):
        latt, freq = lattices[i], frequencies[i]

        idx_latt = l_elts.index(latt)
        idx_freq = l_elts.index(freq)
        assert idx_freq - idx_latt == 1

        idx_of_section_changes.append(idx_latt - 2 * n_secs_before_current)
        n_secs_before_current += 1

        l_elts.pop(idx_freq)
        l_elts.pop(idx_latt)
    idx_of_section_changes.pop(0)
    idx_of_section_changes.append(len(l_elts))

    return l_elts, {'lattices': lattices, 'frequencies': frequencies,
                    'indices': idx_of_section_changes,
                    'n_sections': n_sections}
