#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np


class accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, filename):
        """Create class and structure by calling read_datafile."""
        self.filename = filename
        self.n_elements = 500
        # TODO: handle cases were there the number of elements in the line
        # is different from 500
        self.structure = np.empty((self.n_elements), dtype=object)
        self.read_datafile()

    def read_datafile(self):
        """Read datafile and create structure."""
        i = 0

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
                if(line[0] == 'FIELD_MAP'):
                    self.structure[i] = FIELD_MAP(line)
                    i += 1

                elif(line[0] == 'DRIFT'):
                    self.structure[i] = DRIFT(line, i)
                    i += 1


class DRIFT():
    """Linear drift."""

    def __init__(self, line, i):
        """Add a drift to structure."""
        # First, check validity of input
        self.n_attributes = len(line) - 1
        if((self.n_attributes != 2) and
           (self.n_attributes != 3) and
           (self.n_attributes != 5)):
            raise IOError(
                'Wrong number of arguments for DRIFT element at position '
                + str(i))

        self.element_pos = i

        self.L = line[1]
        self.R = line[2]

        if(self.n_attributes >= 3):
            self.R_y = line[3]
            self.n_attributes = 3

        if(self.n_attributes == 5):
            self.R_x_shift = line[4]
            self.R_y_shift = line[5]
            self.n_attributes = 5

    def info(self):
        """
        Output information on the element.

        Should match the corresponding line in the .dat file.
        """
        if(self.n_attributes == 2):
            print('DRIFT ' + self.L + ' ' + self.R)

        elif(self.n_attributes == 3):
            print('DRIFT ' + self.L + ' ' + self.R + ' ' + self.R_y)

        elif(self.n_attributes == 5):
            print('DRIFT ' + self.L + ' ' + self.R + ' ' + self.R_y +
                  ' ' + self.R_x_shift + ' ' + self.R_y_shift)


class FIELD_MAP():
    """Field map."""

    def __init__(self, line):
        """Add a field map to the structure."""
        # TODO check input validity
        self.geom = line[1]
        self.L = line[2]
        self.theta_i = line[3]
        self.R = line[4]
        self. k_b = line[5]
        self.k_e = line[6]
        self.K_i = line[7]
        self.K_a = line[8]
        self.FileName = line[9]
        self.P = line[10]
