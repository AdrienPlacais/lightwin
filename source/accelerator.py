#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import os.path
import numpy as np
import tracewin_interface as tw
import elements
import helper
from electric_field import load_field_map_file
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
        self.dat_filepath = dat_filepath
        self.project_folder = os.path.dirname(dat_filepath)
        self.n_elements = 39
        # TODO: handle cases were there the number of elements in the line
        # is different from 39

        # Load dat file and clean it up (remove comments, etc)
        self.dat_file_content = tw.load_dat_file(dat_filepath)

        # Create empty list of elements and fill it
        self.list_of_elements = []
        self._create_structure()
        self._load_filemaps()

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

    def _load_filemaps(self):
        """Assign filemaps paths and load them."""
        # Get folder of all field maps
        for line in self.dat_file_content:
            if line[0] == 'FIELD_MAP_PATH':
                field_map_folder = line[1]

        field_map_folder = os.path.dirname(self.dat_filepath) \
            + field_map_folder[1:]

        for elt in self.list_of_elements:
            if 'field_map_file_name' in vars(elt):
                elt.field_map_file_name = field_map_folder \
                    + '/' + elt.field_map_file_name
                load_field_map_file(elt, elt.acc_field)

    def _create_structure(self):
        """Create structure using the loaded dat file."""
        # @TODO Implement lattice
        # Dictionnary linking element name with correct sub-class
        subclasses_dispatcher = {
            'QUAD': elements.Quad,
            'DRIFT': elements.Drift,
            'FIELD_MAP': elements.FieldMap,
            'CAVSIN': elements.CavSin,
            'SOLENOID': elements.Solenoid,
        }
        to_be_implemented = ['SPACE_CHARGE_COMP', 'FREQ', 'FIELD_MAP_PATH',
                             'LATTICE', 'END']
        # TODO Maybe some non-elements such as FREQ or LATTICE would be better
        # off another file/module

        # We look at each element in dat_file_content, and according to the
        # value of the 1st column string we create the appropriate Element
        # subclass and store this instance in list_of_elements
        try:
            list_of_elements = [subclasses_dispatcher[elem[0]](elem)
                                for elem in self.dat_file_content if elem[0]
                                not in to_be_implemented]
        except KeyError:
            print('Warning, an element from filepath was not recognized.')

        self.list_of_elements = list_of_elements

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
            print(data_out, elt)
            if elt.name == 'FIELD_MAP':
                assert key is not None, 'A key must be provided as attribute ' \
                    + 'is a dict.'
                subclass_attributes = vars(elt)

                if data_out is not None:
                    print('a')
                    data_tmp = np.copy(subclass_attributes[attribute][key])
                    data_out = add_data(data_out, data_tmp, attribute)
                else:
                    print('b', subclass_attributes, attribute, key)
                    data_out = np.copy(subclass_attributes[attribute][key])
            else:
                data_out = add_data(data_out, np.NaN, attribute)

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
        return out
