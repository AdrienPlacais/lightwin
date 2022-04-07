#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021.

@author: placais
"""
import os.path
import numpy as np
import tracewin_interface as tw
import helper
import transport
import particle
from constants import FLAG_PHI_ABS, E_MEV, METHOD, E_rest_MeV, FLAG_PHI_S_FIT
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
        dat_filecontent, list_of_elements = tw.load_dat_file(dat_filepath)
        list_of_elements, sections, lattices, freqs =\
            self._sections_lattices(list_of_elements)

        self.elements = {
            'n': len(list_of_elements),
            'list': list_of_elements,
            'l_lattices': lattices,
            'l_sections': sections,
            }

        tw.load_filemaps(dat_filepath, dat_filecontent,
                         self.elements['l_sections'], freqs)
        tw.give_name(self.elements['list'])

        self.files = {
            'project_folder': os.path.dirname(dat_filepath),
            'dat_filecontent': dat_filecontent,
            }

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., E_MEV, n_steps=last_idx,
                                       synchronous=True, reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'cumul': np.full((last_idx+1, 2, 2), np.NaN),
            'indiv': np.full((last_idx+1, 2, 2), np.NaN),
            'first_calc?': True,
            }
        self.transf_mat['indiv'][0, :, :] = np.eye(2)

        # Check that LW and TW computes the phases in the same way
        self._check_consistency_phases()

    def _sections_lattices(self, list_of_elements):
        """Gather elements by section and lattice."""
        list_of_elements, dict_struct = \
            self._prepare_sections_and_lattices(list_of_elements)

        lattices = []
        sections = []
        j = 0
        for i in range(dict_struct['n_sections']):
            lattices = []
            n_lattice = dict_struct['lattices'][i].n_lattice
            while j < dict_struct['indices'][i]:
                lattices.append(list_of_elements[j:j+n_lattice])
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
                    elt.idx['element'] = list_of_elements.index(elt)
            shift_lattice += j + 1
        lattices = []
        for sec in sections:
            lattices += sec
        return list_of_elements, sections, lattices, dict_struct['frequencies']

    def _prepare_sections_and_lattices(self, list_of_elements):
        """
        Save info on the accelerator structure.

        In particular: in every section, the number of elements per lattice,
        the rf frequency of cavities, the position of the delimitations between
        two sections.
        """
        lattices = [
            lattice
            for lattice in list_of_elements
            if type(lattice) is elements.Lattice
            ]
        frequencies = [
            frequency
            for frequency in list_of_elements
            if type(frequency) is elements.Freq
            ]
        n_sections = len(lattices)
        assert n_sections == len(frequencies)

        idx_of_section_changes = []
        n_of_sections_before_this_one = 0
        for i in range(n_sections):
            latt, freq = lattices[i], frequencies[i]

            idx_latt = list_of_elements.index(latt)
            idx_freq = list_of_elements.index(freq)
            assert idx_freq - idx_latt == 1

            idx_of_section_changes.append(idx_latt
                                          - 2 * n_of_sections_before_this_one)
            n_of_sections_before_this_one += 1

            list_of_elements.pop(idx_freq)
            list_of_elements.pop(idx_latt)
        idx_of_section_changes.pop(0)
        idx_of_section_changes.append(len(list_of_elements))

        return list_of_elements, {'lattices': lattices,
                                  'frequencies': frequencies,
                                  'indices': idx_of_section_changes,
                                  'n_sections': n_sections}

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
            flags_absolute.append(cav.acc_field.absolute_phase_flag)

        if FLAG_PHI_ABS and False in flags_absolute:
            print('Warning: you asked LW a simulation in absolute phase,',
                  'while there is at least one cavity in relative phase in',
                  "the .dat file used by TW. Results won't match if there",
                  'are faulty cavities.\n')
        elif not FLAG_PHI_ABS and True in flags_absolute:
            print('Warning: you asked LW a simulation in relative phase,',
                  'while there is at least one cavity in absolute phase in',
                  "the .dat file used by TW. Results won't match if there",
                  'are faulty cavities.\n')

    def compute_transfer_matrices(self, elements=None, transfer_data=True,
                                  fit=False, l_norm=[], l_phi_0=[]):
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        elements : list of Elements, optional
            List of elements from which you want the transfer matrices. Default
            is None.
        """
        if elements is None:
            elements = self.elements['list']

        l_r_zz = []
        l_W_kin = [self.synch.energy['kin_array_mev'][elements[0].idx['s_in']]]
        l_phi_abs = [self.synch.phi['abs_array'][elements[0].idx['s_in']]]

        # Compute transfer matrix and acceleration (gamma) in each element
        if METHOD in ['RK', 'leapfrog']:
            for elt in elements:
                if elt.info['nature'] == 'FIELD_MAP' \
                        and elt.info['status'] != 'failed':
                    if fit and elt.info['status'] == 'compensate':
                        d_fit = {'flag': True, 'norm': l_norm.pop(0),
                                 'phi': l_phi_0.pop(0)}
                    else:
                        d_fit = {'flag': False}

                    kwargs = elt.set_cavity_parameters(
                            self.synch, l_phi_abs[-1], l_W_kin[-1], d_fit)
                    l_r_zz_elt, l_W_kin_elt, l_phi_rel_elt, _ = \
                        elt.compute_transfer_matrix(l_W_kin[-1], **kwargs)

                else:
                    kwargs = None
                    l_r_zz_elt, l_W_kin_elt, l_phi_rel_elt, _ = \
                        elt.compute_transfer_matrix(l_W_kin[-1])

                l_r_zz.append(l_r_zz_elt)
                l_phi_abs_elt = [l_phi_abs[-1] + phi_rel
                                 for phi_rel in l_phi_rel_elt]
                l_phi_abs += l_phi_abs_elt
                l_W_kin += l_W_kin_elt

                idx = range(elt.idx['s_in'] + 1, elt.idx['s_out'] + 1)

                if transfer_data:
                    self.synch.transfer_data(elt, l_W_kin_elt, l_phi_abs_elt)
                    elt.tmat['matrix'] = l_r_zz_elt
                    self.transf_mat['indiv'][idx] = l_r_zz_elt
                    if kwargs is not None:
                        elt.acc_field.transfer_data(**kwargs)

            idxs = [elements[0].idx['s_in'], elements[-1].idx['s_out'] + 1]

            flattened_r_zz = np.concatenate(l_r_zz)
            n_r_zz = len(range(idxs[0], idxs[1]))
            cumul_r_zz = np.full((n_r_zz, 2, 2), np.NaN)

            if idxs[0] == 0:
                cumul_r_zz[0, :, :] = np.eye(2)
            else:
                cumul_r_zz[0, :, :] = self.transf_mat['cumul'][idxs[0], :, :]

            for i in range(1, n_r_zz):
                cumul_r_zz[i, :, :] = flattened_r_zz[i-1, :, :] \
                    @ cumul_r_zz[i-1, :, :]
            if transfer_data:
                self.transf_mat['cumul'][idxs[0]:idxs[1]] = cumul_r_zz

        elif METHOD == 'transport':
            print('computer_transfer_matrices: no MT computation with ',
                  'transport method.')
            transport.transport_particle(self, self.synch)

        return cumul_r_zz, l_W_kin, l_phi_abs

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
