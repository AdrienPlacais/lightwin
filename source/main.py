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
    - Replace flag phi_abs/phi_rel by a three positions switch: synch/abs/rel?
"""
import os
import time
from datetime import timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from constants import I_MILLI_A
import accelerator as acc
import debug
import helper
import emittance
import tracewin_interface as tw
import fault_scenario as mod_fs

# Select .dat file
Tk().withdraw()

FILEPATH = "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
# FILEPATH = "../data/JAEA/JAEA_ADS_026.dat"
if FILEPATH == "":
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])

# =============================================================================
# Fault compensation
# =============================================================================
failed_cav = [
    # 29, 108,
    # 35,
    205,
    # 155, 157, 167,
    # 295, 307,
    # 355,
    # 395,
    # 521, 523, 525, 527,
    # 583
]

FLAG_FIX = True
SAVE_FIX = True

# =============================================================================
# Outputs
# =============================================================================
PLOTS = [
    "energy",
    # "phase",
    "cav",
    # "emittance",
    # "twiss",
    "enveloppes",
]
PLOT_TM = False

SAVES = [
    "energy phase and mt",
    "Vcav and phis",
]

DICT_PLOTS_PRESETS = {
    "energy": [["energy", "energy_err", "struct"], 21],
    "phase": [["abs_phase", "abs_phase_err", "struct"], 22],
    "cav": [["v_cav_mv", "field_map_factor", "phi_s_deg", "struct"], 23],
    "emittance": [["eps_w", "eps_zdelta", "struct"], 24],
    "twiss": [["alpha_w", "beta_w", "gamma_w"], 25],
    "enveloppes": [["envel_pos_w", "envel_ener_w", "struct"], 26],
}

DICT_SAVES = {
    "energy phase and mt": helper.save_energy_phase_tm,
    "Vcav and phis": helper.save_vcav_and_phis,
}

if abs(I_MILLI_A) > 1e-10:
    helper.printc("main.py warning: ", opt_message="I_MILLI_A is not zero,"
                  + "but LW does not take space charge forces into account.")


# =============================================================================
# Start
# =============================================================================
start_time = time.monotonic()
FILEPATH = os.path.abspath(FILEPATH)

# Reference linac
ref_linac = acc.Accelerator(FILEPATH, "Working")
ref_linac.compute_transfer_matrices()
for plot in PLOTS:
    debug.compare_with_tracewin(ref_linac, x_dat="s",
                                y_dat=DICT_PLOTS_PRESETS[plot][0],
                                fignum=DICT_PLOTS_PRESETS[plot][1])

# Broken linac
broken_linac = acc.Accelerator(FILEPATH, "Broken")
fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav)
fail.transfer_phi0_from_ref_to_broken()
broken_linac.compute_transfer_matrices()
for plot in PLOTS:
    debug.compare_with_tracewin(broken_linac, x_dat="s",
                                y_dat=DICT_PLOTS_PRESETS[plot][0],
                                fignum=DICT_PLOTS_PRESETS[plot][1])

if FLAG_FIX:
    fail.fix_all()
    broken_linac.compute_transfer_matrices()
    for plot in PLOTS:
        debug.compare_with_tracewin(broken_linac, x_dat="s",
                                    y_dat=DICT_PLOTS_PRESETS[plot][0],
                                    fignum=DICT_PLOTS_PRESETS[plot][1])

if PLOT_TM:
    debug.plot_transfer_matrices(ref_linac, ref_linac.transf_mat["cumul"])

#     for save in SAVES:
#         DICT_SAVES[save](lin)

if SAVE_FIX:
    tw.save_new_dat(broken_linac, FILEPATH)

# =============================================================================
# End
# =============================================================================
end_time = time.monotonic()
print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))
fail.evaluate_fit_quality()
