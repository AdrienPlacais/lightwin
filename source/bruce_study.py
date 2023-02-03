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
import core.accelerator as acc
import core.fault_scenario as mod_fs
from util import debug
from util import helper
import util.tracewin_interface as tw

if __name__ == '__main__':
    # Select .dat file
    FILEPATH = "../data/JAEA/JAEA_ADS_026.dat"

    # =========================================================================
    # Fault compensation
    # =========================================================================
    FLAG_FIX = True
    SAVE_FIX = True
    FLAG_TW = False

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

    failed_8 = [[25]]
    wtf_8 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[12, 14, 23, 27, 29, 31]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

    failed_9 = [[40, 42, 44, 46, 48]]
    wtf_9 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 3, 'l': 2, 'manual list': [[6, 8, 10, 12, 14,
                                              23, 25, 27, 29, 31,
                                              57, 59, 61, 63, 65,
                                              74, 76, 78, 80, 82]],
             'objective': ['w_kin', 'phi_abs_array'],  # , 'mismatch factor'],
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
    l_failed = [failed_1]
    l_wtf = [wtf_1]

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
        print(f"\n\nElapsed time: {timedelta(seconds=end_time - start_time)}")
        delta_t = timedelta(seconds=end_time - start_time)
        ranking = fail.evaluate_fit_quality(delta_t)
        helper.printd(ranking, header='Fit evaluation')
        data = tw.output_data_in_tw_fashion(lin)

        if SAVE_FIX:
            helper.printc("main warning: ", opt_message="if studying several "
                          "linacs, the .dat of first fix will be replaced by "
                          "last one.")
            filepath = os.path.join(lin.get('out_lw'),
                                    os.path.basename(FILEPATH))
            os.makedirs(lin.get('out_lw'))
            tw.save_new_dat(lin, filepath, data, ranking)

    for lin in linacs:
        for plot in PLOTS:
            kwargs = debug.DICT_PLOT_PRESETS[plot]
            kwargs['linac_ref'] = linacs[0]
            debug.compare_with_tracewin(lin, **kwargs)

    if FLAG_TW:
        lin = linacs[-1]
        ini_path = FILEPATH.replace(".dat", ".ini")
        os.makedirs(lin.get('out_tw'))
        kwargs = {'path_cal': lin.get('out_tw'),
                  'dat_file': lin.get('dat_filepath')}
        tw.run_tw(lin, ini_path, **kwargs)
