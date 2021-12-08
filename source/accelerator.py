#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import os.path
import numpy as np
import elements
import helper
from constants import m_MeV
from electric_field import load_field_map_file
import transport


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, e_0_mev, f_mhz, dat_filepath):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.list_of_elements.

        Parameters
        ----------
        e_0_mev: float
            Initial beam energy in MeV.
        dat_filepath: string
            Path to file containing the structure.
        """
        self.dat_filepath = dat_filepath
        self.project_folder = os.path.dirname(dat_filepath)
        self.n_elements = 39
        # TODO: handle cases were there the number of elements in the line
        # is different from 39

        self.e_0_mev = e_0_mev
        self.f_mhz_bunch = f_mhz

        # Load dat file and clean it up (remove comments, etc)
        self.dat_file_content = []
        self._load_dat_file()

        # Create empty list of elements and fill it
        self.list_of_elements = []
        self._create_structure()
        self._load_filemaps()

        # Longitudinal transfer matrix of the first to the i-th element:
        self.transfer_matrix_cumul = np.full((1, 2, 2), np.NaN)
        self.flag_first_calculation_of_transfer_matrix = True

    def _load_dat_file(self):
        """Load the dat file and convert it into a list of lines."""
        # Load and read data file
        with open(self.dat_filepath) as file:
            for line in file:
                # Remove trailing whitespaces
                line = line.strip()

                # We check that the current line is not empty or that it is not
                # reduced to a comment only
                if(len(line) == 0 or line[0] == ';'):
                    continue

                # Remove any trailing comment
                line = line.split(';')[0]
                line = line.split()

                self.dat_file_content.append(line)

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
            list_of_elements = [subclasses_dispatcher[elem[0]](elem, self.f_mhz_bunch)
                                for elem in self.dat_file_content if elem[0]
                                not in to_be_implemented]
        except KeyError:
            print('Warning, an element from filepath was not recognized.')

        self.list_of_elements = list_of_elements

    def _complementary_assignation(self, e_0_mev):
        """Define Elements attributes that are dependent to each others."""
        entry = 0.
        out = 0.

        for i in range(self.n_elements):
            elt = self.list_of_elements[i]
            out += elt.length_m

            elt.pos_m['abs'] = elt.pos_m['rel'] + entry
            entry = out

        # Initial energy:
        self.list_of_elements[0].energy['e_array_mev'][0] = e_0_mev
        self.list_of_elements[0].energy['gamma_array'][0] = \
            helper.mev_to_gamma(e_0_mev, m_MeV)

    def compute_transfer_matrices(self, method):
        """Compute the transfer matrices of Accelerator's elements."""
        for elt in self.list_of_elements:
            elt.init_solver_settings(method)

        self._complementary_assignation(self.e_0_mev)

        gamma_out = self.list_of_elements[0].energy['gamma_array'][0]

        # Compute transfer matrix and acceleration (gamma) in each element
        if method in ['RK', 'leapfrog']:
            for element in self.list_of_elements:
                element.energy['gamma_array'][0] = gamma_out
                element.energy['e_array_mev'][0] = \
                    helper.gamma_to_mev(gamma_out, m_MeV)
                element.compute_transfer_matrix()

                if self.flag_first_calculation_of_transfer_matrix:
                    self.transfer_matrix_cumul = element.transfer_matrix
                    self.flag_first_calculation_of_transfer_matrix = False

                else:
                    np.vstack((self.transfer_matrix_cumul,
                               element.transfer_matrix))

                element.energy['e_array_mev'] = helper.gamma_to_mev(
                    element.energy['gamma_array'], m_MeV)
                gamma_out = element.energy['gamma_array'][-1]

            transfer_matrix_indiv = np.expand_dims(np.eye(2), axis=0)
            self.transfer_matrix_indiv = np.vstack((transfer_matrix_indiv,
                                                    self.get_from_elements(
                                                        'transfer_matrix')
                                                    ))

        elif method == 'transport':
            transport.transport_beam(self)
            transfer_matrix_indiv = self.transfer_matrix_indiv

        self.transfer_matrix_cumul = \
            helper.individual_to_global_transfer_matrix(
                self.transfer_matrix_indiv)

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
            }
        out = None
        data_nature = str(type(list_of_keys[attribute]))
        for elt in self.list_of_elements:
            out = dict_data_getter[data_nature](out, elt)
        return out
