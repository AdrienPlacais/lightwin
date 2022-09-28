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
import accelerator as acc
import debug
import helper
import emittance
import tracewin_interface as tw
import fault_scenario as mod_fs

# Select .dat file
Tk().withdraw()

# FILEPATH = "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
FILEPATH = "../data/JAEA/JAEA_ADS_026.dat"
if FILEPATH == "":
    FILEPATH = askopenfilename(filetypes=[("TraceWin file", ".dat")])

# =============================================================================
# Fault compensation
# =============================================================================
manual_list = [7, 15, 17, 25, 27]
failed_cav = [
    # 35,
    # 155, 157,
    # 295, 307,
    355,
    # 395,
    # 521, 523, 525, 527,
    # 583
]
manual_list = [
    [25, 27, 37, 45, 47],
    # [145, 147, 165, 167, 175, 177, 185, 187],
    # [285, 287, 297, 305, 315, 317, 325, 327],
    # [345, 347, 357, 365, 367],
    # [385, 387, 397, 399, 401],
    # [493, 495, 497, 499, 507, 509, 511, 513,
    # 535, 537, 539, 541, 549, 551, 553, 555],
    # [579, 581, 591, 593, 595, 597]
]

FLAG_FIX = False
SAVE_FIX = False

# =============================================================================
# Outputs
# =============================================================================
PLOTS = [
    "energy",
    "phase",
    "cav",
    "emittance",
    "twiss",
    "enveloppes",
]
PLOT_TM = False
PHASE_SPACE = False
TWISS = False

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
    "energy phase and mt": lambda lin: helper.save_energy_phase_tm(lin),
    "Vcav and phis": lambda lin: helper.save_vcav_and_phis(lin),
}

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

# # Broken linac
# broken_linac = acc.Accelerator(FILEPATH, "Broken")
# fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav)
# fail.transfer_phi0_from_ref_to_broken()
# broken_linac.compute_transfer_matrices()
# for plot in PLOTS:
#     debug.compare_with_tracewin(broken_linac, x_dat="s",
#                                 y_dat=DICT_PLOTS_PRESETS[plot][0],
#                                 fignum=DICT_PLOTS_PRESETS[plot][1])

# if FLAG_FIX:
#     fail.prepare_compensating_cavities_of_all_faults(manual_list)
#     fail.fix_all()
#     broken_linac.compute_transfer_matrices()
#     for plot in PLOTS:
#         debug.compare_with_tracewin(broken_linac, x_dat="s",
#                                     y_dat=DICT_PLOTS_PRESETS[plot][0],
#                                     fignum=DICT_PLOTS_PRESETS[plot][1])
# Broken linac but with proper cavities status
# fail.update_status_of_cavities_that_compensate(manual_list)
# broken_linac.compute_transfer_matrices()
# linacs = [ref_linac, broken_linac]

# for lin in linacs:
#     lin.compute_transfer_matrices()

if PLOT_TM:
    debug.plot_transfer_matrices(ref_linac, ref_linac.transf_mat["cumul"])

#     for save in SAVES:
#         DICT_SAVES[save](lin)

#         if SAVE_FIX:
#             tw.save_new_dat(broken_linac, FILEPATH)
#         # Redo this whole loop with a fixed linac
#         linacs.append(broken_linac)

# =============================================================================
# End
# =============================================================================
end_time = time.monotonic()
print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))

DEBUT_ELLIPSE = False
if DEBUT_ELLIPSE:
    fig, axx = helper.create_fig_if_not_exist(3, [221, 222])
    lala = ref_linac.elements["list"][35].idx["s_in"]
    for (i, j) in zip(range(2), [0, lala]):#ref_linac.synch.z["abs_array"].shape[0] - 1]):
        for lin in [ref_linac]:#, broken_linac]:
            debug.plot_ellipse_emittance(axx[i], lin, j)

# data_ref = tw.output_data_in_tw_fashion(ref_linac)
# data_fixed = tw.output_data_in_tw_fashion(broken_linac)
# fault_info = fail.faults['l_obj'][0].info

DEBUG_EMITT = False
import matplotlib.pyplot as plt
if DEBUG_EMITT:
    import numpy as np

    # OK now let's try some trucs.
    # sigma matrix at the entry of the linac
    sigma_zdelta = np.array(([2.9511603e-06, -1.9823111e-07],
                             [-1.9823111e-07, 7.0530641e-07]))

    # Total transfer matrix
    ref = np.loadtxt("/home/placais/LightWin/data/faultcomp22/working/results/Longitudinalemittance(πdegMeV).txt")
    fig, ax = helper.create_fig_if_not_exist(13, [111])
    ax = ax[0]
    ax.plot(ref[:, 0], ref[:, 1], label='Ref')
    ax.legend()

    emittance.calc_emittance_from_tw_transf_mat(ref_linac, sigma_zdelta)
    plt.show()
