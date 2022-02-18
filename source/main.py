#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021

@author: placais
"""
import os
import time
from datetime import timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import accelerator as acc
import debug
import helper
import emittance
import transport
import fault
import tracewin_interface as tw
# =============================================================================
# User inputs
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
start_time = time.monotonic()
ref_linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH, 'Working')

broken_linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH, 'Broken')
failed_cav = [25]
manual_list = [15, 17, 27, 35, 37]
strategy = 'manual'
objective = 'energy'


# fault.apply_faults(broken_linac, failed_cav)
basic_fault = fault.fault_scenario(ref_linac, broken_linac)
basic_fault.break_at(failed_cav)


for lin in [ref_linac, broken_linac]:
    for method in ['RK']:
        lin.compute_transfer_matrices(method)

# =============================================================================
# Output options
# =============================================================================
        PLOT_TM = False
        PLOT_ENERGY = True
        PLOT_CAV = True
        PHASE_SPACE = False
        TWISS = False
        SAVE_MT_AND_ENERGY = False
        SAVE_VCAV_AND_PHIS = False

        if PLOT_TM:
            debug.plot_transfer_matrices(lin, lin.transf_mat['cumul'])

        if PLOT_ENERGY:
            debug.compare_energies(lin)

        if PLOT_CAV:
            debug.plot_vcav_and_phis(lin)

        if PHASE_SPACE:
            debug.compare_phase_space(lin)

        if TWISS:
            twiss = emittance.transport_twiss_parameters(lin, ALPHA_Z, BETA_Z)
            emittance.plot_twiss(lin, twiss)

        if SAVE_MT_AND_ENERGY:
            helper.save_full_mt_and_energy_evolution(lin)

        if SAVE_VCAV_AND_PHIS:
            helper.save_vcav_and_phis(lin)


# basic_fault.fix(strategy, objective, manual_list)
tw.save_new_dat(broken_linac, FILEPATH)

if PLOT_ENERGY:
    debug.compare_energies(broken_linac)
if PLOT_CAV:
    debug.plot_vcav_and_phis(broken_linac)
if PLOT_TM:
    debug.plot_transfer_matrices(broken_linac,
                                 broken_linac.transf_mat['cumul'])

end_time = time.monotonic()
print('\n\nElapsed time:', timedelta(seconds=end_time - start_time))
