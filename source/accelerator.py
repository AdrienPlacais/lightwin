#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np
import elements as elem
import helper


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in a numpy array self.structure.
        """
        self.n_elements = 5000
        # TODO: handle cases were there the number of elements in the line
        # is different from 5000
        self.structure = np.empty((self.n_elements), dtype=object)

    def create_struture_from_dat_file(self, filename):
        """
        Read datafile and create structure.

        Parameters
        ----------
        filename: string
            Path to the .dat file to study.
        """
        self.filename = filename

        # To keep track of the number of elements in the list of elements
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
                    self.structure[i] = elem.Drift(line, i)
                    i += 1

                elif(element_name == 'QUAD'):
                    self.structure[i] = elem.Quad(line, i)
                    i += 1

                elif(element_name == 'FIELD_MAP'):
                    self.structure[i] = elem.FieldMap(line, i, self.filename)
                    i += 1

                elif(element_name == 'DRIFT'):
                    self.structure[i] = elem.Drift(line, i)
                    i += 1

                elif(element_name in list_of_non_elements):
                    continue

                else:
                    msg = "Element not yet implemented: "
                    opt_msg = line[0] + '\t\t (i=' + str(i) + ')'
                    helper.printc(msg, color='info', opt_message=opt_msg)

        # Resize array of elements
        self.n_elements = i
        self.structure = np.resize(self.structure, (self.n_elements))

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

        for i in range(idx_min, idx_max):
            self.structure[i].show_element_info()
