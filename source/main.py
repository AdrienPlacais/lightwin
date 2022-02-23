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
import tracewin_interface as tw
import fault

# =============================================================================
# User inputs
# =============================================================================
# TODO: direct import of this parameters from the .ini file
# TODO: handle different particles
# Kinetic beam energy in MeV
E_MEV = 16.6

# Current in mA
I_MILLI_A = 0.0

# Bunch frequency in MHz
F_MHZ = 176.1

# Input normalized rms emittance (pi.mm.mrad)
EMIT_Z_Z_PRIME = 0.27
# Longitudinal rms emittance (pi.deg.MeV)
emit_pw = emittance.mm_mrad_to_deg_mev(EMIT_Z_Z_PRIME, F_MHZ)

# Input Twiss parameters. Not modified by EMIT_Z_Z_PRIME
ALPHA_Z = 0.1389194
BETA_Z = 2.1311577  # mm/pi.mrad
BETA_W = 71.215849  # deg/pi.MeV

# Select .dat file
Tk().withdraw()
FILEPATH = "../data/work_field_map/work_field_map.dat"
if FILEPATH == "":
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])

# =============================================================================
# Fault compensation
# =============================================================================
failed_cav = [25]
manual_list = [15, 17, 27, 35, 37]
STRATEGY = "manual"
OBJECTIVE = "phase"
FLAG_FIX = False
SAVE_FIX = False

# =============================================================================
# Outputs
# =============================================================================
PLOTS = [
    "energy",
    "phase",
    "cav",
    ]
PLOT_TM = False
PHASE_SPACE = False
TWISS = False
SAVE_MT_AND_ENERGY = False
SAVE_VCAV_AND_PHIS = False

start_time = time.monotonic()
# =============================================================================
# Start
# =============================================================================
FILEPATH = os.path.abspath(FILEPATH)
ref_linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH, "Working")
broken_linac = acc.Accelerator(E_MEV, F_MHZ, FILEPATH, "Broken")

basic_fault = fault.fault_scenario(ref_linac, broken_linac)
basic_fault.break_at(failed_cav)

PRESETS = {
    "energy": [["energy", "energy_err", "struct"], 21],
    "phase": [["abs_phase", "abs_phase_err", "struct"], 22],
    "cav": [["v_cav_mv", "phi_s_deg", "struct"], 23],
    }

for lin in [ref_linac, broken_linac]:
    for method in ["RK"]:
        lin.compute_transfer_matrices(method)

        for plot in PLOTS:
            debug.compare_with_tracewin(lin, x_dat="s", y_dat=PRESETS[plot][0],
                                        fignum=PRESETS[plot][1])
        if PLOT_TM:
            debug.plot_transfer_matrices(lin, lin.transf_mat["cumul"])

        if PHASE_SPACE:
            debug.compare_phase_space(lin)

        if TWISS:
            twiss = emittance.transport_twiss_parameters(lin, ALPHA_Z, BETA_Z)
            emittance.plot_twiss(lin, twiss)

        if SAVE_MT_AND_ENERGY:
            helper.save_full_mt_and_energy_evolution(lin)

        if SAVE_VCAV_AND_PHIS:
            helper.save_vcav_and_phis(lin)

if FLAG_FIX:
    basic_fault.fix(STRATEGY, OBJECTIVE, manual_list)

    if SAVE_FIX:
        tw.save_new_dat(broken_linac, FILEPATH)

    for plot in PLOTS:
        debug.compare_with_tracewin(broken_linac, x_dat="s",
                                    y_dat=PRESETS[plot][0],
                                    fignum=PRESETS[plot][1])

    if PLOT_TM:
        debug.plot_transfer_matrices(broken_linac,
                                     broken_linac.transf_mat["cumul"])

# =============================================================================
# End
# =============================================================================
end_time = time.monotonic()
print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))
