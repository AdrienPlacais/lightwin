#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:33:39 2022.

@author: placais
"""

import os
from copy import deepcopy
import time
from datetime import timedelta
import accelerator as acc
import debug
import helper
import tracewin_interface as tw
import fault_scenario as mod_fs

if __name__ == '__main__':
    # Select .dat file
    FILEPATH = "../data/JAEA/JAEA_ADS_026.dat"

    # =========================================================================
    # Fault compensation
    # =========================================================================
    FLAG_FIX = True
    SAVE_FIX = True

    failed_1 = [[12]]
    wtf_1 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[6, 8, 10, 14, 23]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_2 = [[14]]
    wtf_2 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[8, 10, 12, 23, 25]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_3 = [[125]]
    wtf_3 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[114, 116, 127, 129, 131]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_4 = [[127]]
    wtf_4 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[114, 116, 125, 129, 131]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_5 = [[129]]
    wtf_5 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[116, 125, 127, 131, 133]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_6 = [[131]]
    wtf_6 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[116, 125, 127, 129, 133]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_7 = [[133]]
    wtf_7 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[116, 125, 127, 129, 131]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}
    # =========================================================================
    # Outputs
    # =========================================================================
    PLOTS = [
        "energy",
        # "phase",
        # "cav",
        # "emittance",
        # "twiss",
        # "envelopes",
    ]

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

    # =========================================================================
    # Start
    # =========================================================================
    FILEPATH = os.path.abspath(FILEPATH)

    # Reference linac
    ref_linac = acc.Accelerator(FILEPATH, "Working")
    results = ref_linac.elts.compute_transfer_matrices()
    ref_linac.save_results(results, ref_linac.elts)

    linacs = [ref_linac]

    # Broken linac
    # lsq_info = None
    # l_failed = [failed_1, failed_2, failed_3, failed_4, failed_5, failed_6,
    #             failed_7]
    # l_wtf = [wtf_1, wtf_2, wtf_3, wtf_4, wtf_5, wtf_6, wtf_7]
    l_failed = [failed_5]
    l_wtf = [wtf_5]

    for [wtf, failed] in zip(l_wtf, l_failed):
        start_time = time.monotonic()
        lin = acc.Accelerator(FILEPATH, "Broken " + str(failed[0][0]))
        fail = mod_fs.FaultScenario(ref_linac, lin, failed, wtf=wtf)
        linacs.append(deepcopy(lin))

        if FLAG_FIX:
            fail.fix_all()
            results = lin.elts.compute_transfer_matrices()
            lin.save_results(results, lin.elts)

        linacs.append(lin)

        # Output some info onthe quality of the fit
        end_time = time.monotonic()
        print("\n\nElapsed time:", timedelta(seconds=end_time - start_time))
        delta_t = timedelta(seconds=end_time - start_time)
        ranking = fail.evaluate_fit_quality(delta_t)
        helper.printd(ranking, header='Fit evaluation')
        data = tw.output_data_in_tw_fashion(lin)

        if SAVE_FIX:
            helper.printc("main warning: ", opt_message="if studying several "
                          "linacs, the .dat of first fix will be replaced by "
                          "last one.")
            filepath = FILEPATH[:-4] + '_fixed_' + str(failed[0][0]) + '.dat'
            tw.save_new_dat(lin, filepath, data, ranking)

    for lin in linacs:
        for plot in PLOTS:
            debug.compare_with_tracewin(lin, x_str="z_abs",
                                        l_y_str=DICT_PLOTS_PRESETS[plot][0],
                                        fignum=DICT_PLOTS_PRESETS[plot][1])
