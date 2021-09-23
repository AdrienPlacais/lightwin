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

# =============================================================================
# User inputs
# =============================================================================
# TODO: direct import of this parameters from the .ini file
# TODO: handle different particles
# Kinetic beam energy in MeV
E_MeV = 1.

# Current in mA
I_mA = 0.5

# Bunch frequency in MHz
f_MHz = 575.

# Select .dat file
Tk().withdraw()
filepath = '/home/placais/LightWin/data/dummy.dat'
# filepath = '/home/placais/TraceWin/work_compensation/work_compensation.dat'
# filepath = askopenfilename(filetypes=[("TraceWin file", ".dat")])


# =============================================================================
# End of user inputs
# =============================================================================
LINAC = acc.Accelerator(E_MeV, I_mA, f_MHz)
LINAC.create_struture_from_dat_file(filepath)
R_zz_tot = LINAC.compute_transfer_matrix_and_gamma(idx_min=0, idx_max=8)
print("Transfer matrix:\n", R_zz_tot)
