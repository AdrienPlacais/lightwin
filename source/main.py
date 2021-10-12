#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename

import accelerator as acc
# import transfer_matrices
from debug import plot_error_on_transfer_matrices_components


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

plot_error_on_transfer_matrices_components(LINAC)
