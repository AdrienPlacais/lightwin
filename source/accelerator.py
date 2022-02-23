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
        in the list self.list_of_elements.
        The data such as the synch phase or the beam energy will be stored in
        the self.synch Particle object.
        """
        self.name = name
        self.project_folder = os.path.dirname(dat_filepath)

        # Load dat file and clean it up (remove comments, etc)
        self.dat_filecontent, self.list_of_elements = \
            tw.load_dat_file(dat_filepath)
        self.n_elements = len(self.list_of_elements)

        self.transf_mat = {
            'cumul': np.expand_dims(np.eye(2), axis=0),
            'indiv': np.expand_dims(np.eye(2), axis=0),
            'first_calc?': True,
            }

        self.prepared = False

    def _prepare_compute_transfer_matrices(self, method):
        """
        Define Elements attributes that are dependent to each others.

        In particular, absolute position of elements' I/O, energy at first
        element entrance, arrays, solvers.

        Parameters
        ----------
        method: string
            Resolution method. 'RK' (Runge-Kutta) or 'leapfrog' for analytical
            transfer matrices. 'transport' for calculation by transporting
            particles through the line.
        """
        pos_in = 0.
        pos_out = 0.
        idx_in = 0
        idx_out = 0

        for i in range(self.n_elements):
            elt = self.list_of_elements[i]
            elt.init_solver_settings(method)

            pos_out += elt.length_m
            idx_out = idx_in + elt.solver_transf_mat.n_steps

            elt.pos_m['abs'] = elt.pos_m['rel'] + pos_in
            elt.idx['in'] = idx_in
            elt.idx['out'] = idx_out
            elt.idx['elt'] = i

            pos_in = pos_out
            idx_in = idx_out

        # Define some arrays to the proper size
        e_0_mev = 16.6
        omega_0 = 2e6 * np.pi * 176.1
        self.synch = particle.Particle(0., e_0_mev, omega_0,
                                       n_steps=idx_out,
                                       synchronous=True)
        self.prepared = True

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
            elements = self.list_of_elements

        if not self.prepared:
            self._prepare_compute_transfer_matrices(method)

        # Compute transfer matrix and acceleration (gamma) in each element
        if method in ['RK', 'leapfrog']:
            for elt in elements:
                elt.compute_transfer_matrix(self.synch)

            self.transf_mat['indiv'] = np.vstack((
                self.transf_mat['indiv'],
                self.get_from_elements('transfer_matrix')))

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
        list_of_keys = vars(self.list_of_elements[0])
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
        for elt in self.list_of_elements:
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
