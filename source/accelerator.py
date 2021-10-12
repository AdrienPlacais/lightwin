#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:54:19 2021

@author: placais
"""
import numpy as np
import elements as elem
import helper
import transfer_matrices
from constants import c, m_MeV


class Accelerator():
    """Class holding the list of the accelerator's elements."""

    def __init__(self, E_MeV, I_mA, f_MHz):
        """
        Create Accelerator object.

        The different elements constituting the accelerator will be stored
        in a numpy array self.structure.

        Parameters
        ----------
        E_MeV: float
            Beam energy in MeV.
        I_mA: float
            Beam current in mA.
        f_MHz: float
            Bunch frequency in MHz.
        """
        self.n_elements = 41
        # TODO: handle cases were there the number of elements in the line
        # is different from 5000

        # Beam arrays; by default, they are considered as constant along the
        # line
        self.E_MeV = np.full((self.n_elements + 1), E_MeV)
        self.I_mA = np.full((self.n_elements), I_mA)
        self.f_MHz = np.full((self.n_elements), f_MHz)

        # Array containing gamma at each in/out element
        self.gamma = np.full((self.n_elements + 1), np.NaN)
        self.gamma[0] = 1. + E_MeV / m_MeV

        # Common to every element
        self.elements_nature = np.full((self.n_elements), np.NaN, dtype=object)
        self.elements_resume = np.full((self.n_elements), np.NaN, dtype=object)
        self.z_transfer_func = np.full((self.n_elements), np.NaN, dtype=object)

        # Elements length, apertures
        self.L_mm = np.full((self.n_elements), np.NaN)
        self.L_m = np.full((self.n_elements), np.NaN)
        self.R = np.full((self.n_elements), np.NaN)

        # Specific to drift
        self.R_y = np.full((self.n_elements), np.NaN)
        self.R_x_shift = np.full((self.n_elements), np.NaN)
        self.R_y_shift = np.full((self.n_elements), np.NaN)

        # Specific to QUAD
        self.G = np.full((self.n_elements), np.NaN)
        self.Theta = np.full((self.n_elements), np.NaN)
        self.G3_over_u3 = np.full((self.n_elements), np.NaN)
        self.G4_over_u4 = np.full((self.n_elements), np.NaN)
        self.G5_over_u5 = np.full((self.n_elements), np.NaN)
        self.G6_over_u6 = np.full((self.n_elements), np.NaN)
        self.GFR = np.full((self.n_elements), np.NaN)

        # Specific to SOLENOID
        self.B = np.full((self.n_elements), np.NaN)

        # Specific to FIELD_MAP
        self.geom = np.full((self.n_elements), np.NaN)
        self.theta_i = np.full((self.n_elements), np.NaN)
        self.k_b = np.full((self.n_elements), np.NaN)
        self.k_e = np.full((self.n_elements), np.NaN)
        self.K_i = np.full((self.n_elements), np.NaN)
        self.K_a = np.full((self.n_elements), np.NaN)
        self.FileName = np.full((self.n_elements), np.NaN, dtype=object)
        self.P = np.full((self.n_elements), np.NaN)
        # Import
        self.nz = np.full((self.n_elements), np.NaN, dtype=int)
        self.zmax = np.full((self.n_elements), np.NaN)
        self.Norm = np.full((self.n_elements), np.NaN)
        self.Fz_array = np.full((self.n_elements), np.NaN, dtype=object)

        # Specific to SPACE_CHARGE_COMP
        self.k = np.full((self.n_elements), np.NaN)

        # Init empty structures and transfer matrix function
        self.structure = np.empty((self.n_elements), dtype=str)
        self.resume = np.empty((self.n_elements), dtype=str)
        self.transfer_matrix_z = transfer_matrices.dummy

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

        # TODO: LATTICE will be needed someday
        list_of_non_elements = ['FIELD_MAP_PATH', 'LATTICE', 'END', ]
        current_f_MHz = self.f_MHz

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
                    elem.add_drift(self, line, i)
                    i += 1

                elif(element_name == 'QUAD'):
                    elem.add_quad(self, line, i)
                    i += 1

                elif(element_name == 'FIELD_MAP'):
                    elem.add_field_map(self, line, i, self.filename,
                                       current_f_MHz)
                    i += 1

                elif(element_name == 'DRIFT'):
                    elem.add_drift(self, line, i)
                    i += 1

                elif(element_name == 'SOLENOID'):
                    elem.add_solenoid(self, line, i)
                    i += 1

                elif(element_name == 'SPACE_CHARGE_COMP'):
                    # TODO: check if i += 1 should be after the current mod
                    self.z_transfer_func[i] = transfer_matrices.not_an_element
                    i += 1
                    self.I_mA[i:] = self.I_mA[i] * (1 - line[1])

                elif(element_name == 'FREQ'):
                    # We change frequency from next element up to the end of
                    # the line
                    self.f_MHz[i+1:] = line[1]

                elif(element_name in list_of_non_elements):
                    continue

                else:
                    msg = "Element not yet implemented: "
                    opt_msg = line[0] + '\t\t (i=' + str(i) + ')'
                    helper.printc(msg, color='info', opt_message=opt_msg)

    def show_elements_info(self, idx_min=0, idx_max=0):
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
            print(self.elements_resume[i])

    def compute_transfer_matrix_and_gamma(self, idx_min=0, idx_max=0):
        """
        Compute the longitudinal transfer matrix of the line.

        Optional indexes allow one to compute the transfer matrix of some
        elements of the line only.

        Parameters
        ----------
        gamma: float
            Lorentz factor of the particle.
        idx_min: int, optional
            Position of first element.
        idx_max: int, optional
            Position of last element.
        """
        # TODO: handle acceleration of particle
        # TODO: precompute gamma at each element entrance. Required in order
        # to compute transfer matrice of a subset of elements.
        if(idx_max == 0):
            idx_max = self.n_elements

        R_zz_tot = np.eye(2, 2)

        for i in range(idx_min, idx_max):
            if(self.elements_nature[i] == 'FIELD_MAP'):
                # FIXME harmonize with other elements
                # TODO Check this Ncell truc.
                R_zz_next, E_out_MeV = \
                    transfer_matrices.z_field_map_electric_field(
                        self.E_MeV[i], self.f_MHz[i], self.Fz_array[i],
                        self.k_e[i], self.theta_i[i], 2, self.nz[i],
                        self.zmax[i])

                self.E_MeV[i+1:] = E_out_MeV
                beta = np.sqrt((1. + E_out_MeV / m_MeV)**2 - 1.) /    \
                    (1. + E_out_MeV / m_MeV)
                # TODO Functions MeV to beta, beta to MeV, MeV to gamma, etc
                # TODO Even better: a setter to update all these arrays
                # together.
                self.gamma[i+1] = 1. / np.sqrt(1. - beta**2)

            else:
                # print(self.L_m[i])
                # print(self.gamma[i])
                R_zz_next = self.z_transfer_func[i](self.L_m[i], self.gamma[i])
                self.gamma[i+1] = self.gamma[i]

            R_zz_tot = np.matmul(R_zz_tot, R_zz_next)

        return R_zz_tot
