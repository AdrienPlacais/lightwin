#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:40:22 2022

@author: placais
"""


class Section():
    """Class to hold a Section composed of several lattices."""

    def __init__(self, freq_rf_mhz, n_cell):
        self.elements = {
            'list': [],
            'list_of_lattices': [],
            }
        self.freq_rf_mhz = freq_rf_mhz
        self.n_cell = n_cell

    def unzip_lattice(self):
        """Put all elements of all the lattices in elements['list']."""
        list_of_elts = [elt
                        for lattice in self.elements['list_of_lattices']
                        for elt in lattice.list_of_elements
                        ]
        self.elements['list'] = list_of_elts
