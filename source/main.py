#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 11:32:12 2021.

@author: placais

General TODO list:
    - Easy way to plot TW's solutions.
    - Better way to compare various LW solutions?
    - Reprofile to check where could be speed up.
        Interpolation function could be replaced by pre-interpolated array?
    - Recheck non synch particles.
    - Method should be in constants.
    - Replace flag phi_abs/phi_rel by a three positions switch: synch/abs/rel?
"""
import os
import time
from datetime import timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import cProfile
import pstats
import accelerator as acc
import debug
import helper
import emittance
import tracewin_interface as tw
import fault_scenario as mod_fs


# =============================================================================
# User inputs
# =============================================================================
# TODO: direct import of this parameters from the .ini file
# TODO: handle different particles
# Current in mA
I_MILLI_A = 0.0

# Input normalized rms emittance (pi.mm.mrad)
EMIT_Z_Z_PRIME = 0.27
# Longitudinal rms emittance (pi.deg.MeV)
# emit_pw = emittance.mm_mrad_to_deg_mev(EMIT_Z_Z_PRIME, F_MHZ)

# Input Twiss parameters. Not modified by EMIT_Z_Z_PRIME
ALPHA_Z = 0.1389194
BETA_Z = 2.1311577  # mm/pi.mrad
BETA_W = 71.215849  # deg/pi.MeV

# Select .dat file
Tk().withdraw()
# FILEPATH = ""
# FILEPATH = "../data/work_field_map/work_field_map.dat"
FILEPATH = "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
if FILEPATH == "":
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])

# =============================================================================
# Fault compensation
# =============================================================================
# failed_cav = [25]
manual_list = [7, 15, 17, 25, 27]
failed_cav = [
    # 35,
    155#, 157,
    # 295, 307,
    # 355,
    # 395,
    # 521, 523, 525, 527,
    # 583
]
manual_list = [
    # [25, 27, 37, 45, 47],
    [145, 147, 165, 167, 175, 177, 185, 187],
    # [285, 287, 297, 305, 315, 317, 325, 327],
    # [345, 347, 357, 365, 367],
    # [385, 387, 397, 399, 401],
    # [493, 495, 497, 499, 507, 509, 511, 513,
     # 535, 537, 539, 541, 549, 551, 553, 555],
    # [579, 581, 591, 593, 595, 597]
]

FLAG_FIX = True
SAVE_FIX = True

FLAG_PROFILE = False
# =============================================================================
# Outputs
# =============================================================================
PLOTS = [
    "energy",
    "phase",
    "cav",
]
PLOT_TM = True
PHASE_SPACE = False
TWISS = False

SAVES = [
    # "MT and energy",
    # "Vcav and phis",
]
SAVE_MT_AND_ENERGY = False
SAVE_VCAV_AND_PHIS = False

start_time = time.monotonic()
# =============================================================================
# Start
# =============================================================================
FILEPATH = os.path.abspath(FILEPATH)
ref_linac = acc.Accelerator(FILEPATH, "Working")
broken_linac = acc.Accelerator(FILEPATH, "Broken")

fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav, manual_list)

DICT_PLOTS_PRESETS = {
    "energy": [["energy", "energy_err", "struct"], 21],
    "phase": [["abs_phase", "abs_phase_err", "struct"], 22],
    "cav": [["v_cav_mv", "field_map_factor", "phi_s_deg", "struct"], 23],
}

DICT_SAVES = {
    "MT and energy": lambda lin: helper.save_full_mt_and_energy_evolution(lin),
    "Vcav and phis": lambda lin: helper.save_vcav_and_phis(lin),
}

linacs = [ref_linac, broken_linac]
for lin in linacs:
    lin.compute_transfer_matrices()

    # FIXME find a way to make this part cleaner
    if lin.name == 'Working':
        fail.transfer_phi0_from_ref_to_broken()

    for plot in PLOTS:
        debug.compare_with_tracewin(lin, x_dat="s",
                                    y_dat=DICT_PLOTS_PRESETS[plot][0],
                                    fignum=DICT_PLOTS_PRESETS[plot][1])
    if PLOT_TM:
        debug.plot_transfer_matrices(lin, lin.transf_mat["cumul"])

    if PHASE_SPACE:
        debug.compare_phase_space(lin)

    if TWISS:
        twiss = emittance.transport_twiss_parameters(lin, ALPHA_Z, BETA_Z)
        emittance.plot_twiss(lin, twiss)

    for save in SAVES:
        DICT_SAVES[save](lin)

    # broken_linac.name is changed to "Fixed" or "Poorly fixed" in fix
    if FLAG_FIX and lin.name == "Broken":
        if FLAG_PROFILE:
            pr = cProfile.Profile()
            pr.enable()
            pr.runcall(fail.fix_all)
            pr.disable()
        else:
            fail.fix_all()

        if SAVE_FIX:
            tw.save_new_dat(broken_linac, FILEPATH)
        # Redo this whole loop with a fixed linac
        linacs.append(broken_linac)

# =============================================================================
# End
# =============================================================================
end_time = time.monotonic()
print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))

data_ref = tw.output_data_in_tw_fashion(ref_linac)
data_fixed = tw.output_data_in_tw_fashion(broken_linac)

if FLAG_PROFILE:
    ps = pstats.Stats(pr).sort_stats('tottime')
    ps.print_stats(.05)

    ps2 = pstats.Stats(pr).sort_stats('cumtime')
    ps.print_stats(.05)
