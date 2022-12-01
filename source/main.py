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
    - raise error when the failed_cav is not a list of list (when manual)
"""
import os
from copy import deepcopy
import time
from datetime import timedelta
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from constants import I_MILLI_A
import accelerator as acc
import debug
import helper
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
FLAG_FIX = True
FLAG_TRY_OPTI_METHODS = True
SAVE_FIX = False

failed_cav = [
    25,
    # 155, 157, #167, # 295, 307, # 355, # 395, # 521, 523, 525, 527, # 583
]

wtf_pso = {'opti method': 'PSO',
           'strategy': 'k out of n',
           'k': 6, 'l': 2, 'manual list': [],
           'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
           'position': 'end_mod', 'phi_s fit': False}

wtf_lsq = {'opti method': 'least_squares',
           'strategy': 'k out of n',
            'k': 2, 'l': 2, 'manual list': [],
            # 'k': 6, 'l': 2, 'manual list': [],
           'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
           'position': 'end_mod', 'phi_s fit': True}
WHAT_TO_FIT = {
    # 'opti method': 'least_squares',
    'opti method': 'PSO',
    # =========================================================================
    # strategy: manual
    # You must provide a list of lists of broken cavities, and the
    # corresponding list of lists of compensating cavities. Broken cavities in
    # a sublist are fixed together with the provided sublist of compensating
    # cavities.
    # =========================================================================
    # 'strategy': 'manual',
    'manual list': [
        # [145, 147, 165, 175, 177]
    ],
    # =========================================================================
    # strategy: k out of n
    # You must provide a list of broken cavities, and the number of
    # compensating cavities per faulty cavity. Close broken cavities are
    # gathered and fixed together.
    # =========================================================================
    'strategy': 'k out of n',
    'k': 6,
    # =========================================================================
    # strategy: l neighboring lattices
    # You must provide a list of broken cavities, and the number of
    # compensating lattices per faulty cavity. Close broken cavities are
    # gathered and fixed together.
    # =========================================================================
    # 'strategy': 'l neighboring lattices',
    'l': 2,
    # =========================================================================
    #     What should we fit?
    # =========================================================================
    'objective': [
        'w_kin',
        'phi_abs_array',
        # 'eps_zdelta', 'beta_zdelta', 'gamma_zdelta',  # 'alpha_zdelta',
        # 'M_11', 'M_12', 'M_22',  # 'M_21',
        'mismatch factor',
    ],
    # =========================================================================
    #     Where should we evaluate objective?
    # =========================================================================
    'position': 'end_mod',
    # 'position': '1_mod_after',
    # 'position': 'both',
    'phi_s fit': True,
}

# =============================================================================
# Outputs
# =============================================================================
PLOTS = [
    "energy",
    # "phase",
    "cav",
    # "emittance",
    # "twiss",
    "envelopes",
    # "mismatch factor",
]
PLOT_TM = False

SAVES = [
    "energy phase and mt",
    "Vcav and phis",
]

DICT_PLOTS_PRESETS = {
    "energy": [["w_kin", "w_kin_err", "struct"], 21],
    "phase": [["phi_abs_array", "phi_abs_array_err", "struct"], 22],
    "cav": [["v_cav_mv", "k_e", "phi_s", "struct"], 23],
    "emittance": [["eps_w", "eps_zdelta", "struct"], 24],
    "twiss": [["alpha_w", "beta_w", "gamma_w"], 25],
    "envelopes": [["envelope_pos_w", "envelope_energy_w", "struct"], 26],
    "mismatch factor": [["mismatch factor", "struct"], 27],
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
FILEPATH = os.path.abspath(FILEPATH)

# Reference linac
ref_linac = acc.Accelerator(FILEPATH, "Working")
results = ref_linac.elts.compute_transfer_matrices()
ref_linac.save_results(results, ref_linac.elts)

linacs = [ref_linac]

# Broken linac
lsq_info = None
# for wtf in []:
# for wtf in [wtf_lsq, wtf_pso]:
for wtf in [wtf_lsq]:
    start_time = time.monotonic()
    lin = acc.Accelerator(FILEPATH, "Broken")
    fail = mod_fs.FaultScenario(ref_linac, lin, failed_cav,
                                wtf=wtf, l_info_other_sol=lsq_info)
    # linacs.append(deepcopy(lin))

    if FLAG_FIX:
        fail.fix_all()
        results = lin.elts.compute_transfer_matrices()
        lin.save_results(results, lin.elts)

        if wtf == wtf_lsq:
            lsq_info = [f.info for f in fail.faults['l_obj']]

    linacs.append(lin)

    # Output some info onthe quality of the fit
    end_time = time.monotonic()
    print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))
    delta_t = timedelta(seconds=end_time - start_time)
    ranking = fail.evaluate_fit_quality(delta_t)
    helper.printd(ranking, header='Fit evaluation')

    if SAVE_FIX:
        helper.printc("main warning: ", opt_message="if studying several "
                      + "linacs, the .dat of first fix will be replaced by "
                      + "last one.")
        tw.save_new_dat(lin, FILEPATH)

for lin in linacs:
    for plot in PLOTS:
        debug.compare_with_tracewin(lin, x_str="z_abs",
                                    l_y_str=DICT_PLOTS_PRESETS[plot][0],
                                    fignum=DICT_PLOTS_PRESETS[plot][1])
if PLOT_TM:
    debug.plot_transfer_matrices(ref_linac, ref_linac.transf_mat["cumul"])
