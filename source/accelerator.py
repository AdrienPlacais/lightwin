#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np
import elements as elements
import helper
from constants import m_MeV


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, E_0_MeV, dat_filepath):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in the list self.list_of_elements.

        Parameters
        ----------
        E_0_MeV: float
            Initial beam energy in MeV.
        dat_filepath: string
            Path to file containing the structure.
        """
        self.dat_filepath = dat_filepath
        self.n_elements = 39
        # TODO: handle cases were there the number of elements in the line
        # is different from 39

        # Load dat file and cleam it up (remove comments, etc)
        self.dat_file_content = []
        self.load_dat_file()

        # Create empty list of elements and fill it
        self.list_of_elements = []
        self.create_structure()
        self.complementary_assignation(E_0_MeV)

        # Longitudinal transfer matrix of the first to the i-th element:
        self.transfer_matrix_cumul = np.full((1, 2, 2), np.NaN)
        self.flag_first_calculation_of_transfer_matrix = True

    def load_dat_file(self):
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

    def create_structure(self):
        """Create structure using the loaded dat file."""
        # Dictionnary linking element name with correct sub-class
        subclasses_dispatcher = {
            'QUAD': elements.Quad,
            'DRIFT': elements.Drift,
            'FIELD_MAP': elements.FieldMap,
            'CAVSIN': elements.CavSin,
            'SOLENOID': elements.Solenoid,
            'SPACE_CHARGE_COMP': elements.NotAnElement,
            'FREQ': elements.NotAnElement,
            'FIELD_MAP_PATH': elements.NotAnElement,
            'LATTICE': elements.NotAnElement,
            'END': elements.NotAnElement,
        }
        to_be_implemented = ['SPACE_CHARGE_COMP', 'FREQ', 'FIELD_MAP_PATH',
                             'LATTICE', 'END']
        # @TODO Maybe some non-elements such as FREQ or LATTICE would be better
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

    def complementary_assignation(self, E_0_MeV):
        """Define Elements attributes that are dependent to each others."""
        entry = 0.
        out = self.list_of_elements[0].length_m

        for i in range(0, self.n_elements):
            element = self.list_of_elements[i]
            element.absolute_position_m = np.array(([entry, out]))
            entry = out
            out += element.length_m

        # Initial energy:
        self.list_of_elements[0].energy_array_mev[0] = E_0_MeV
        self.list_of_elements[0].gamma_array[0] = helper.mev_to_gamma(E_0_MeV,
                                                                      m_MeV)

    def compute_transfer_matrices(self):
        """Compute the transfer matrices of Accelerator's elements."""
        # TODO Maybe it would be better to compute transfer matrices at
        # elements initialization?
        gamma_out = self.list_of_elements[0].gamma_array[0]

        for element in self.list_of_elements:
            element.gamma_array[0] = gamma_out
            element.energy_array_mev[0] = helper.gamma_to_mev(gamma_out,
                                                              m_MeV)
            element.compute_transfer_matrix()

            if self.flag_first_calculation_of_transfer_matrix:
                self.transfer_matrix_cumul = element.transfer_matrix
                self.flag_first_calculation_of_transfer_matrix = False

            else:
                np.vstack((self.transfer_matrix_cumul,
                           element.transfer_matrix))

            # TODO better of elsewhere?
            element.energy_array_mev = helper.gamma_to_mev(element.gamma_array,
                                                           m_MeV)

            gamma_out = element.gamma_array[-1]

        self.transfer_matrix_cumul = \
            helper.individual_to_global_transfer_matrix(
                self.transfer_matrix_cumul
                )

    def get_attribute_of_elements(self, truc):
        out = [element.truc for element in self.list_of_elements]
        return out
