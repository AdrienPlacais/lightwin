#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""

# import numpy as np
# import pandas as pd
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import accelerator as acc
import transfer_matrices
# =============================================================================
# User inputs
# =============================================================================
# TODO: direct import of this parameters from the .ini file
# TODO: handle different particles
# Kinetic beam energy in MeV
E_MeV = 16.6

# Current in mA
I_mA = 4.

# Bunch frequency in MHz
f_MHz = 176.1

# Select .dat file
Tk().withdraw()
# filepath = '/home/placais/LightWin/data/dummy.dat'
# filepath = '/home/placais/TraceWin/work_compensation/work_compensation.dat'
filepath = '/home/placais/TraceWin/work_field_map/work_field_map.dat'
# filepath = askopenfilename(filetypes=[("TraceWin file", ".dat")])


# =============================================================================
# End of user inputs
# =============================================================================
LINAC = acc.Accelerator(E_MeV, I_mA, f_MHz)
LINAC.create_struture_from_dat_file(filepath)
# LINAC.show_elements_info(0, 40)
R_zz_tot = LINAC.compute_transfer_matrix_and_gamma(idx_min=0, idx_max=39)
print("Transfer matrix:\n", R_zz_tot)
print("E = ", LINAC.E_MeV[-1])

# i = 37
# lala, bu = transfer_matrices.z_field_map_electric_field(
#                         18.793905, LINAC.f_MHz[i], LINAC.Fz_array[i],
#                         LINAC.k_e[i], LINAC.theta_i[i], 2, LINAC.nz[i],
#                         LINAC.zmax[i])
# print('MT', lala)
# print('E_out', bu, 'MeV')
