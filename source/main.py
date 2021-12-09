#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""

from tkinter import Tk
from tkinter.filedialog import askopenfilename
import accelerator as acc
import debug
import helper

# =============================================================================
# User inputs5
# =============================================================================
# TODO: direct import of this parameters from the .ini file
# TODO: handle different particles
# Kinetic beam energy in MeV
E_MEV = 16.6

# Current in mA
I_MILLI_A = 0.

# Bunch frequency in MHz
F_MHZ = 176.1

# Select .dat file
Tk().withdraw()
# FILEPATH = '/home/placais/TraceWin/work_compensation/work_compensation.dat'
FILEPATH = '/home/placais/TraceWin/work_field_map/work_field_map.dat'
# FILEPATH = '/home/placais/TraceWin/work_cavsin/work_cavsin.dat'
if FILEPATH == '':
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])


# =============================================================================
# End of user inputs
# =============================================================================
linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH)

for method in ['transport']:
    linac.compute_transfer_matrices(method)
    debug.plot_transfer_matrices(linac, linac.transfer_matrix_cumul)
    # debug.compare_energies(linac)

# transport.compute_envelope(linac)

SAVE_MT_AND_ENERGY = False
if SAVE_MT_AND_ENERGY:
    helper.save_full_mt_and_energy_evolution(linac)

SAVE_VCAV_AND_PHIS = False
if SAVE_VCAV_AND_PHIS:
    helper.save_vcav_and_phis(linac)
