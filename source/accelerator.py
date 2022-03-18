#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import os.path
import numpy as np
import tracewin_interface as tw
import helper
import transport
import particle
from constants import FLAG_PHI_ABS, E_MEV, F_BUNCH_MHZ
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
            'list_lattice': lattices,
            'sections': sections,
            }

        tw.load_filemaps(dat_filepath, dat_filecontent,
                         self.elements['sections'], freqs)
        tw.give_name(self.elements['list'])

        self.files = {
            'project_folder': os.path.dirname(dat_filepath),
            'dat_filecontent': dat_filecontent,
            }

        # Set indexes and absolute position of the different elements
        last_idx = self._set_indexes_and_abs_positions()

        # Create synchronous particle
        omega_0 = 2e6 * np.pi * F_BUNCH_MHZ
        reference = bool(name == 'Working')
        self.synch = particle.Particle(0., E_MEV, omega_0,
                                       n_steps=last_idx, synchronous=True,
                                       reference=reference)

        # Transfer matrices
        self.transf_mat = {
            'cumul': np.expand_dims(np.eye(2), axis=0),
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

        zones = ['LEBT', 'MEBT', 'HEBT']
        for i, sec in enumerate(sections):
            for lattice in sec:
                for elt in lattice:
                    elt.info['zone'] = zones[i]

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
            elt.idx = idx.copy()
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

    def compute_transfer_matrices(self, method, elements=None):
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        method: string
            Resolution method. 'RK' (Runge-Kutta) or 'leapfrog' for analytical
            transfer matrices. 'transport' for calculation by transporting
            particles through the line.
        elements: list of Elements, opt
            List of elements from which you want the transfer matrices.
        """
        if elements is None:
            elements = self.elements['list']

        for elt in elements:
            elt.tmat['solver_param']['method'] = method

        # Compute transfer matrix and acceleration (gamma) in each element
        if method in ['RK', 'leapfrog']:
            for elt in elements:
                elt.compute_transfer_matrix(self.synch)
                idx = [elt.idx['in'] + 1, elt.idx['out'] + 1]
                self.transf_mat['indiv'][idx[0]:idx[1], :, :] = \
                    elt.tmat['matrix']

            self.transf_mat['cumul'] = \
                helper.individual_to_global_transfer_matrix(
                    self.transf_mat['indiv'])

        elif method == 'transport':
            print('computer_transfer_matrices: no MT computation with ',
                  'transport method.')
            transport.transport_particle(self, self.synch)

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
            idx = self.elements_of(nature=elt.info['nature']).index(elt)
        else:
            idx = self.elements['list'].index(elt)

        return idx

    def where_is_section(self, elt):
        """Give the abs position of the element."""
        idx = {
            'absolute_lattice': None,
            'absolute_element': self.elements['list'].index(elt),
            'nested': [(idsec, idlatt, idelt)
                       for (idsec, sec) in enumerate(self.elements['sections'])
                       for (idlatt, latt) in enumerate(sec)
                       for (idelt, elem) in enumerate(latt)
                       if elem is elt][0]
            }
        idx['absolute_lattice'] = idx['nested'][1]
        for n_sec in range(idx['nested'][0]):
            idx['absolute_lattice'] += len(self.elements['sections'][n_sec])

        return idx

    def where_is_this_index(self, idx, showinfo=False):
        """Give an equivalent index."""
        for elt in self.elements['list']:
            if idx in range(elt.idx['in'], elt.idx['out']):
                break
        if showinfo:
            print('Synch index', idx, 'is in:', elt.info)
            print('Synch indexes of this elt:', elt.idx, '\n\n')
        return elt
