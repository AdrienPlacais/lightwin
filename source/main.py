#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""
import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import accelerator as acc
import debug
import helper
import emittance
import transport

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

# Input normalized rms emittance (pi.mm.mrad)
EMIT_Z_Z_PRIME = 0.27
# Longitudinal rms emittance (pi.deg.MeV)
emit_pw = emittance.mm_mrad_to_deg_mev(EMIT_Z_Z_PRIME, F_MHZ)

# Input Twiss parameters. Not modified by EMIT_Z_Z_PRIME
ALPHA_Z = 0.1389194
BETA_Z = 2.1311577      # mm/pi.mrad
BETA_W = 71.215849      # deg/pi.MeV

# Select .dat file
Tk().withdraw()
FILEPATH = '../data/work_field_map/work_field_map.dat'
if FILEPATH == '':
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])

FILEPATH = os.path.abspath(FILEPATH)
# =============================================================================
# End of user inputs
# =============================================================================
linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH)

for method in ['transport']:
    linac.compute_transfer_matrices(method)
    # debug.plot_transfer_matrices(linac, linac.transf_mat['cumul'])
    # debug.compare_energies(linac)
debug.compare_phase_space(linac)

# twiss = emittance.transport_twiss_parameters(linac, ALPHA_Z, BETA_Z)
# emittance.plot_twiss(linac, twiss)

SAVE_MT_AND_ENERGY = False
if SAVE_MT_AND_ENERGY:
    helper.save_full_mt_and_energy_evolution(linac)

SAVE_VCAV_AND_PHIS = False
if SAVE_VCAV_AND_PHIS:
    helper.save_vcav_and_phis(linac)

# =============================================================================
# Checking
# =============================================================================
print('\n\n')
# Why is initial delta_phi different from zero?
delta_z_par = linac.partran_data[0]['z(mm)'][5]
delta_z_lw = linac.particle_list[5].phase_space['z_array'][0] * 1e3
delta_delta_z = delta_z_par - delta_z_lw
print('diff of delta_z between tools:', delta_delta_z, 'ok')

import numpy as np
delta_phi_par = linac.partran_data[0]['Phase(deg)'][5]
delta_phi_lw = np.rad2deg(linac.particle_list[5].phase_space['phi_array_rad'][0])
delta_delta_phi = delta_phi_par - delta_phi_lw
print('diff of delta_phi:', delta_delta_phi)
