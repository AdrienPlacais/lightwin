#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021

@author: placais
"""


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
