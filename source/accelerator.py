#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import os.path
from collections import defaultdict
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
        self.n_elements = 39
        # TODO: handle cases were there the number of elements in the line
        # is different from 39

        # Load dat file and clean it up (remove comments, etc)
        self.dat_filecontent, self.list_of_elements = \
            tw.load_dat_file(dat_filepath)

        self.transf_mat = {
            'cumul': np.expand_dims(np.eye(2), axis=0),
            'indiv': np.expand_dims(np.eye(2), axis=0),
            'first_calc?': True,
            }

        self.fault_scenario = {
            'faults_idx': [],        # List of indexes of faulty cavities
            'faults_cav': [],
            'comp_idx': [],
            'comp_cav': [],          # List of compensating cavity objects
            'strategy': None,        # To determine cav_compensating
            'x0': None,              # Initial parameters for the fit
            'bounds': None,          # Parameters bounds
            'objective_str': None,   # Name of variable that should match
            'objective': None,       # Variable that should match
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
        Return the attribute of all elements.

        Parameters
        ----------
        attribute: string
            Name of the desired attribute.
        key: string, optional
            If attribute is a dict, key must be provided.
        """
        print(attribute, key)
        # Some attributes such as enery hold in/out data: energy at the
        # entrance and at the exit of the element. As energy at the entrance
        # of an element is equal to the energy at the exit of the precedent,
        # we discard all entrance data.
        discard_list = ['pos_m', 'energy']
        if attribute == 'energy':
            return self.synch.energy['kin_array_mev']

        def add_data(data_out, data_tmp, attribute):
            """
            Add data_tmp to data_out.

            This function also removes the first element of data_tmp if
            attribute is in discard_list.
            """
            if attribute in discard_list:
                data_tmp = data_tmp[1:]
                data_out = np.hstack((data_out, data_tmp))

            else:
                data_out = np.vstack((data_out, data_tmp))
            return data_out

        def fun_array(data_out, elt):
            """
            Create an array of required attribute.

            Used when element[attribute] is an array.
            """
            subclass_attributes = vars(elt)

            if data_out is not None:
                data_tmp = np.copy(subclass_attributes[attribute])
                data_out = add_data(data_out, data_tmp, attribute)
            else:
                data_out = np.copy(subclass_attributes[attribute])

            return data_out

        def fun_dict(data_out, elt):
            """
            Create an array of required attribute.

            Used when element[attribute] is dict (eg attribute = 'energy' and
                                                  key = 'gamma_array').
            """
            assert key is not None, 'A key must be provided as attribute ' \
                + 'is a dict.'
            subclass_attributes = vars(elt)

            if data_out is not None:
                data_tmp = np.copy(subclass_attributes[attribute][key])
                data_out = add_data(data_out, data_tmp, attribute)
            else:
                data_out = np.copy(subclass_attributes[attribute][key])

            return data_out

        def fun_rf(data_out, elt):
            """
            Create an array of required attribute.

            Used when element[attribute] is RfField.
            """
            if elt.name == 'FIELD_MAP':
                assert key is not None, 'A key must be provided as attribute '\
                    + 'is a class.'
                subclass_attributes = vars(elt)

                if data_out is not None:
                    data_tmp = np.copy(getattr(subclass_attributes[attribute],
                                               key))
                    data_out = add_data(data_out, data_tmp, attribute)
                else:
                    data_out = np.copy(subclass_attributes[attribute][key])
            else:
                if data_out is not None:
                    data_out = add_data(data_out, np.NaN, attribute)
                else:
                    data_out = np.array([np.NaN])
            return data_out

        def fun_simple(data_out, elt):
            """
            Create an array of required attribute.

            Used when element[attribute] is 'simple': float or a string (eg
            'name').
            """
            subclass_attributes = vars(elt)

            if data_out is not None:
                data_tmp = subclass_attributes[attribute]
                data_out = np.hstack((data_out, data_tmp))
            else:
                data_out = np.array((subclass_attributes[attribute]))

            return data_out

        # Get list of attributes of first element
        list_of_keys = vars(self.list_of_elements[0])

        dict_data_getter = {
            "<class 'numpy.ndarray'>": fun_array,
            "<class 'dict'>": fun_dict,
            "<class 'float'>": fun_simple,
            "<class 'str'>": fun_simple,
            "<class 'electric_field.RfField'>": fun_rf,
            }
        out = None
        data_nature = str(type(list_of_keys[attribute]))
        for elt in self.list_of_elements:
            out = dict_data_getter[data_nature](out, elt)
        if key in ['v_cav_mv', 'phi_s_deg']:   # FIXME
            out = out[:, 0]
        return out


    def blabla(self, attribute, key=None):
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
            "<class 'numpy.ndarray'>": lambda data_new: data_new,
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
