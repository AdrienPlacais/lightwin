#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np
import elements as elem


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
                    self.structure[i] = elem.Drift(line, i)
                    i += 1

                elif(element_name == 'QUAD'):
                    self.structure[i] = elem.Quad(line, i)
                    i += 1

                elif(element_name == 'FIELD_MAP'):
                    self.structure[i] = elem.FieldMap(line, i)
                    i += 1

                elif(element_name == 'DRIFT'):
                    self.structure[i] = elem.Drift(line, i)
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
