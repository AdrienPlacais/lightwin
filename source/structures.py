#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, filename):
        """
        Create class and structure by calling create_struture_from_dat_file.

        Parameters
        ----------
        filename: string
            Path to the .dat file to study.

        """
        self.filename = filename
        self.n_elements = 5000
        # TODO: handle cases were there the number of elements in the line
        # is different from 5000
        self.structure = np.empty((self.n_elements), dtype=object)
        self.create_struture_from_dat_file()

    def create_struture_from_dat_file(self):
        """Read datafile and create structure."""
        i = 0

        # TODO: LATTICE and FREQ will be needed someday
        list_of_non_elements = ['FIELD_MAP_PATH', 'LATTICE', 'FREQ', 'END', ]

        # Load and read data file
        with open(self.filename) as file:
            for line in file:
                # Remove trailing whitespaces
                line = line.strip()

                # We check that the current line is not empty or that it is not
                # reduced to a comment only
                if(len(line) == 0 or line[0] == ';'):
                    continue

                # We remove any trailing comment
                line = line.split(';')[0]

                # ID element:
                line = line.split()
                element_name = line[0]

                if(element_name == 'DRIFT'):
                    self.structure[i] = Drift(line, i)
                    i += 1

                elif(element_name == 'QUAD'):
                    self.structure[i] = Quad(line, i)
                    i += 1

                elif(element_name == 'FIELD_MAP'):
                    self.structure[i] = FieldMap(line, i)
                    i += 1

                elif(element_name == 'DRIFT'):
                    self.structure[i] = Drift(line, i)
                    i += 1

                elif(element_name in list_of_non_elements):
                    continue

                else:
                    print('Element not yet implemented: ' + line[0])

    def show_all_elements_info(self, idx_min=0, idx_max=0):
        """
        Recursively call info function of all structure's elements.

        Parameters
        ----------
        idx_min: int, optional
            Position of first element to output.
        idx_max: int, optional
            Position of last element to output.
        """
        if(idx_max == 0):
            idx_max = self.n_elements

        for i in range(idx_min, idx_max + 1):
            self.structure[i].show_element_info()


class Element():
    """Super class holding methods and properties common to all elements."""

    def __init__(self, line, i):
        """
        Initialize what is common to all ELEMENTs.

        Attributes
        ----------
        n_attributes: integer
            The number of attributes in the .dat file.
        element_pos: integer
            Position of the element. Should match TraceWin's.
        resume: string
            Resume of the element properties. Should match the corresponding
            line in the .dat file, at the exception of the first character
            that is the elemet position.
        """
        self.n_attributes = len(line) - 1
        self.element_pos = i
        self.resume = [str(self.element_pos)] + line
        self.resume = ' '.join(self.resume)

    def show_element_info(self):
        """
        Output information on the element.

        Should match the corresponding line in the .dat file.
        """
        print(self.resume)


class Drift(Element):
    """Linear drift."""

    def __init__(self, line, i):
        """Add a drift to structure."""
        super().__init__(line, i)

        # First, check validity of input
        if((self.n_attributes != 2) and
           (self.n_attributes != 3) and
           (self.n_attributes != 5)):
            raise IOError(
                'Wrong number of arguments for DRIFT element at position '
                + str(self.element_pos))

        self.L = line[1]
        self.R = line[2]

        try:
            self.R_y = line[3]
            self.R_x_shift = line[4]
            self.R_y_shift = line[5]
        except IndexError:
            pass


class Quad(Element):
    """Quadrupole."""

    def __init__(self, line, i):
        """Add a quadrupole to structure."""
        super().__init__(line, i)

        # First, check validity of input
        if((self.n_attributes < 3) or
           (self.n_attributes > 9)):
            raise IOError(
                'Wrong number of arguments for QUAD element at position '
                + str(self.element_pos))

        self.L = line[1]
        self.G = line[2]
        self.R = line[3]

        try:
            self.Theta = line[4]
            self.G3_over_u3 = line[5]
            self.G4_over_u4 = line[6]
            self.G5_over_u5 = line[7]
            self.G6_over_u6 = line[8]
            self.GFR = line[9]
        except IndexError:
            pass


class FieldMap(Element):
    """Field map."""

    def __init__(self, line, i):
        """Add a field map to the structure."""
        super().__init__(line, i)

        # TODO check input validity
        if((self.n_attributes < 9) or (self.n_attributes > 10)):
            raise IOError(
                'Wrong number of arguments for FIELD_MAP element at position '
                + str(self.element_pos))

        self.geom = line[1]
        self.L = line[2]
        self.theta_i = line[3]
        self.R = line[4]
        self. k_b = line[5]
        self.k_e = line[6]
        self.K_i = line[7]
        self.K_a = line[8]
        self.FileName = line[9]

        try:
            self.P = line[10]
        except IndexError:
            pass
