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


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, dat_filepath, name):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.elements['list'].
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent, l_elts = tw.load_dat_file(dat_filepath)
        l_elts, l_secs, l_latts, freqs = _sections_lattices(l_elts)

        self.elements = {'n': len(l_elts),
                         'list': l_elts,
                         'l_lattices': l_latts,
                         'l_sections': l_secs}

        tw.load_filemaps(dat_filepath, dat_filecontent,
                         self.elements['l_sections'], freqs)
        tw.give_name(self.elements['list'])

        self.files = {'project_folder': os.path.dirname(dat_filepath),
                      'dat_filecontent': dat_filecontent,
                      'results_folder':
                          os.path.dirname(dat_filepath) + '/results_lw/'}

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'cumul': np.full((last_idx + 1, 2, 2), np.NaN),
            'indiv': np.full((last_idx + 1, 2, 2), np.NaN),
            'first_calc?': True,
        }
        self.transf_mat['indiv'][0, :, :] = np.eye(2)

        # Check that LW and TW computes the phases in the same way (abs or rel)
        self._check_consistency_phases()

    def _set_indexes_and_abs_positions(self):
        """Init solvers, set indexes and absolute positions of elements."""
        pos = {'in': 0., 'out': 0.}
        idx = {'in': 0, 'out': 0}
        for elt in self.elements['list']:
            elt.init_solvers()

            pos['in'] = pos['out']
            pos['out'] += elt.length_m
            elt.pos_m['abs'] = elt.pos_m['rel'] + pos['in']

            idx['in'] = idx['out']
            idx['out'] += elt.tmat['solver_param']['n_steps']
            elt.idx['s_in'], elt.idx['s_out'] = idx['in'], idx['out']
        return idx['out']

    def _check_consistency_phases(self):
        """Check that both TW and LW use absolute or relative phases."""
        cavities = self.elements_of(nature='FIELD_MAP')
        flags_absolute = []
        for cav in cavities:
            flags_absolute.append(cav.acc_field.phi_0['abs_phase_flag'])

        if FLAG_PHI_ABS and False in flags_absolute:
            print("Warning: you asked LW a simulation in absolute phase,",
                  "while there is at least one cavity in relative phase in",
                  "the .dat file used by TW. Results won't match if there",
                  "are faulty cavities.\n")
        elif not FLAG_PHI_ABS and True in flags_absolute:
            print("Warning: you asked LW a simulation in relative phase,",
                  "while there is at least one cavity in absolute phase in",
                  "the .dat file used by TW. Results won't match if there",
                  "are faulty cavities.\n")

    # TODO Is flag_transfer_data=False equivalent to d_fits['flag']=True?
    def compute_transfer_matrices(self, l_elts=None, flag_transfer_data=True,
                                  d_fits=None):
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        l_elts : list of Elements, optional
            List of elements from which you want the transfer matrices. Default
            is None. In this case, MT calculated for the whole linac.
        flag_transfer_data: bool, optional
            To determine if calculated energies, phases, MTs should be saved.
            Default is True.
        d_fits: dict, optional
            Dict to determine if an optimization is performed. If Yes, the
            accelering fields data is taken from this dict instead of from
            the acc_field objects. Default is {'flag': False}.
        """
        if l_elts is None:
            l_elts = self.elements['list']

        if d_fits is None:
            d_fits = {'flag': False}

        # Index of entry of first element, index of exit of last one
        idx_in = l_elts[0].idx['s_in']
        idx_out = l_elts[-1].idx['s_out'] + 1

        # To store results
        l_phi_s_rad = []
        l_w_kin = [self.synch.energy['kin_array_mev'][idx_in]]
        l_phi_abs = [self.synch.phi['abs_array'][idx_in]]
        l_r_zz_elt = []     # List of numpy arrays
        l_rf_fields = []    # List of dicts

        # Compute transfer matrix and acceleration in each element
        for elt in l_elts:
            phi_abs = l_phi_abs[-1]

            if elt.info['nature'] != 'FIELD_MAP' \
               or elt.info['status'] == 'failed':
                rf_field = None
                elt_results = elt.calc_transf_mat(l_w_kin[-1])

            else:
                if d_fits['flag'] \
                   and elt.info['status'] == 'compensate (in progress)':
                    d_fit_elt = {'flag': True,
                                 'phi': d_fits['l_phi'].pop(0),
                                 'k_e': d_fits['l_k_e'].pop(0)}
                else:
                    d_fit_elt = d_fits

                rf_field = elt.set_cavity_parameters(self.synch, phi_abs,
                                                     l_w_kin[-1], d_fit_elt)
                l_rf_fields.append(rf_field)
                elt_results = elt.calc_transf_mat(l_w_kin[-1], **rf_field)
                l_phi_s_rad.append(elt_results['cav_params']['phi_s_rad'])

            r_zz_elt = [elt_results['r_zz'][i, :, :]
                        for i in range(elt_results['r_zz'].shape[0])]
            l_r_zz_elt.extend(r_zz_elt)
            l_phi_abs_elt = [phi_rel + phi_abs
                             for phi_rel in elt_results['phi_rel']]
            l_phi_abs.extend(l_phi_abs_elt)
            l_w_kin.extend(elt_results['w_kin'].tolist())

            if flag_transfer_data:
                # FIXME ok with new way of dealing phi_s?
                self.transfer_data(elt, elt_results, np.array(l_phi_abs_elt),
                                   rf_field)

        # Compute transfer matrix of l_elts
        n_steps = len(l_w_kin)
        arr_r_zz_cumul = np.full((n_steps, 2, 2), np.NaN)

        # If we are at the start of the linac, initial transf mat is unity
        if idx_in == 0:
            arr_r_zz_cumul[0] = np.eye(2)
        else:
            # Else we take the tm at the start of l_elts
            arr_r_zz_cumul[0] = self.transf_mat['cumul'][idx_in]
            assert ~np.isnan(arr_r_zz_cumul[0]).any(), \
                "Previous transfer matrix was not calculated."

        for i in range(1, n_steps):
            arr_r_zz_cumul[i] = l_r_zz_elt[i - 1] @ arr_r_zz_cumul[i - 1]

        if flag_transfer_data:
            self.transf_mat['cumul'][idx_in:idx_out] = arr_r_zz_cumul

        return arr_r_zz_cumul, l_w_kin, l_phi_abs, l_phi_s_rad, l_rf_fields

    def transfer_data(self, elt, elt_results, phi_abs_elt, rf_field):
        """
        Transfer calculated energies, phases, MTs, etc to proper Objects.

        This function is to be used when NO optimisation is performed.
        """
        idx = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)
        self.synch.transfer_data(elt, elt_results['w_kin'], phi_abs_elt)
        elt.tmat['matrix'] = elt_results['r_zz']
        self.transf_mat['indiv'][idx] = elt_results['r_zz']

        # if elt.info['nature'] == 'FIELD_MAP':
        if elt_results['cav_params'] is not None:
            # print(elt_results['cav_params'])
            elt.acc_field.cav_params = elt_results['cav_params']
            elt.acc_field.phi_0['abs'] = rf_field['phi_0_abs']
            elt.acc_field.phi_0['rel'] = rf_field['phi_0_rel']

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
        list_of_keys = vars(self.elements['list'][0])
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
        for elt in self.elements['list']:
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
            sub_list = self.elements['list']
        list_of = list(filter(lambda elt: elt.info['nature'] == nature,
                              sub_list))
        return list_of

    def where_is(self, elt, nature=False):
        """
        Determine where is elt in list_of_elements.

        If nature = True, elt is the idx-th element of his nature.

        Parameters
        ----------
        elt : Element
            Element you want the position of.
        nature : bool, optional
            Allow to count only the elt's nature (eg QUAD). The default is
            False.

        Returns
        -------
        idx : int
            Position of elt in list_of_elements, or in the list of elements of
            it's nature if nature is True.

        """
        if nature:
            print('Warning, calling where_is with argument nature.')
            idx = self.elements_of(nature=elt.info['nature']).index(elt)
        else:
            print('Warning, where_is should be useless now.')
            idx = self.elements['list'].index(elt)

        return idx

    def where_is_this_index(self, idx, showinfo=False):
        """Give an equivalent index."""
        for elt in self.elements['list']:
            if idx in range(elt.idx['s_in'], elt.idx['s_out']):
                break
        if showinfo:
            print('Synch index', idx, 'is in:', elt.info)
            print('Indexes of this elt:', elt.idx, '\n\n')
        return elt


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

    zones = ['low beta', 'medium beta', 'high beta']
    shift_lattice = 0
    for i, sec in enumerate(sections):
        for j, lattice in enumerate(sec):
            for k, elt in enumerate(lattice):
                elt.info['zone'] = zones[i]
                elt.idx['section'] = [(i, j, k)]
                elt.idx['lattice'] = (j + shift_lattice, k)
                elt.idx['element'] = l_elts.index(elt)
        shift_lattice += j + 1
    lattices = []
    for sec in sections:
        lattices += sec
    return l_elts, sections, lattices, dict_struct['frequencies']
