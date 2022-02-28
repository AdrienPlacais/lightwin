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


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, e_0_mev, f_mhz, dat_filepath, name):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.elements['list'].
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name

        # Load dat file, clean it up (remove comments, etc), load elements
        dat_filecontent, list_of_elements, n_lattice = \
            tw.load_dat_file(dat_filepath)

        self.elements = {
            'n': len(list_of_elements),
            'list': list_of_elements,
            'n_per_lattice': n_lattice,
            'list_lattice': [],
            }
        lattice = []
        for i in range(self.elements['n']):
            lattice.append(self.elements['list'][i])
            if len(lattice) == self.elements['n_per_lattice']:
                self.elements['list_lattice'].append(lattice)
                lattice = []
        if len(lattice) > 0:
            self.elements['list_lattice'].append(lattice)
            print("Warning, the last module added to ",
                  "self.elements['list_lattice'] was not full (", len(lattice),
                  " elements instead of ", self.elements['n_per_lattice'], ")")

        self.files = {
            'project_folder': os.path.dirname(dat_filepath),
            'dat_filecontent': dat_filecontent,
            }

        # Set indexes and absolute position of the different elements
        pos = {'in': 0., 'out': 0.}
        idx = {'in': 0, 'out': 0}
        for elt in self.elements['list']:
            elt.init_solvers()

            pos['in'] = pos['out']
            pos['out'] += elt.length_m
            elt.pos_m['abs'] = elt.pos_m['rel'] + pos['in']

            idx['in'] = idx['out']
            idx['out'] += elt.solver_param_transf_mat['n_steps']
            elt.idx = idx.copy()

        # Create synchronous particle
        omega_0 = 2e6 * np.pi * f_mhz
        self.synch = particle.Particle(0., e_0_mev, omega_0,
                                       n_steps=idx['out'], synchronous=True)

        # Transfer matrices
        self.transf_mat = {
            'cumul': np.expand_dims(np.eye(2), axis=0),
            'indiv': np.full((idx['out']+1, 2, 2), np.NaN),
            'first_calc?': True,
            }
        self.transf_mat['indiv'][0, :, :] = np.eye(2)

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
            elt.solver_param_transf_mat['method'] = method

        # Compute transfer matrix and acceleration (gamma) in each element
        if method in ['RK', 'leapfrog']:
            for elt in elements:
                elt.compute_transfer_matrix(self.synch)
                idx = [elt.idx['in'] + 1, elt.idx['out'] + 1]
                self.transf_mat['indiv'][idx[0]:idx[1], :, :] = \
                    elt.transfer_matrix

            # TODO: only recompute what is necessary
            self.transf_mat['cumul'] = \
                helper.individual_to_global_transfer_matrix(
                    self.transf_mat['indiv'])

        elif method == 'transport':
            print('computer_transfer_matrices: no MT computation with ',
                  'transport method.')
            transport.transport_particle(self, self.synch)

    def get_from_elements(self, attribute, key=None):
        """
        Return attribute from all elements in list_of_elements.

        Parameters
        ----------
        attribute: string
            Name of the desired attribute.
        key: string, optional
            If attribute is a dict, key must be provided.
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
            data_out.append(fun(list_of_keys[attribute], key))

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
        list_of = list(filter(lambda elt: elt.name == nature, sub_list))
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
            idx = self.sub_list(elt.name).index(elt)
        else:
            idx = self.elements['list'].index(elt)

        return idx
