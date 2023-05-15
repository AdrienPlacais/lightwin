#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:33:39 2022.

@author: placais
"""

import os
import logging
from copy import deepcopy
import time
import datetime
import pandas as pd

import config_manager as conf_man
import core.accelerator as acc
import core.fault_scenario as mod_fs
import util.helper as helper
import util.output as output
import util.evaluate as evaluate
import util.tracewin_interface as tw
import visualization.plot as plot
from util.log_manager import set_up_logging


if __name__ == '__main__':
    # Select .dat file
    FILEPATH = "../data/JAEA/JAEA_ADS_026.dat"

    # =========================================================================
    # Fault compensation
    # =========================================================================
    FLAG_FIX = False
    SAVE_FIX = False
    FLAG_TW = False
    FLAG_EVALUATE = False

    failed_0 = [12]
    wtf_0 = {'opti method': 'least_squares',
             'strategy': 'k out of n',
             'k': 5, 'l': 2, 'manual list': [[6, 8, 10, 14, 23]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'scale_objective': [1., 1., 1.],
             'position': 'end_mod', 'phi_s fit': True}

# =============================================================================
# Manual entry of BYR study cases
# =============================================================================
    failed_1 = [[12]]
    wtf_1 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[6, 8, 10, 14, 23]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             # 'scale_objective': [1., 100., 1.],
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
             'scale_objective': [1., 1., 1.],
             'position': 'end_mod', 'phi_s fit': True}

    failed_8 = [[25]]
    wtf_8 = {'opti method': 'least_squares', 'strategy': 'manual',
             'k': 6, 'l': 2, 'manual list': [[12, 14, 23, 27, 29, 31]],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

# =============================================================================
#     FullPer V
# =============================================================================
    failed_9 = [125, 127, 129, 131, 133]
    wtf_9 = {'opti method': 'least_squares', 'strategy': 'k out of n',
             'k': 5, 'l': 2, 'manual list': [0],
             'objective': [
                 'w_kin',
                 # 'phi_abs_array',
                 'mismatch factor'],
             'scale_objective': [1., 1., 1.],
             'position': 'end_mod', 'phi_s fit': True}

# =============================================================================
#     FullPer other
# =============================================================================
    failed_10 = [40, 42, 44, 46, 48]
    wtf_10 = {'opti method': 'least_squares', 'strategy': 'k out of n',
              'k': 5, 'l': 2, 'manual list': [0],
              'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
              'position': 'end_mod', 'phi_s fit': True}

# =============================================================================
# Systematic study of all faults
# =============================================================================
    all_cavs = [6, 8, 10, 12, 14,
                23, 25, 27, 29, 31,
                40, 42, 44, 46, 48,
                57, 59, 61, 63, 65,
                74, 76, 78, 80, 82,
                91, 93, 95, 97, 99,
                108, 110, 112, 114, 116,
                125, 127, 129, 131, 133,
                ]
    wtf_classic = {'opti method': 'least_squares', 'strategy': 'k out of n',
                   'k': 5, 'l': 2, 'manual list': [0],
                   'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
                   'position': 'end_mod', 'phi_s fit': True}

    # =========================================================================
    # Outputs
    # =========================================================================
    PLOTS = [
        "energy",
        "phase",
        "cav",
        "emittance",
        # "twiss",  # TODO
        # "envelopes", # FIXME
        # "transfer matrices", # TODO
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
    PROJECT_FOLDER = os.path.join(
        os.path.dirname(FILEPATH),
        datetime.datetime.now().strftime('%Y.%m.%d_%Hh%M_%Ss_%fms'))
    os.makedirs(PROJECT_FOLDER)
    set_up_logging(logfile_file=os.path.join(PROJECT_FOLDER, 'lightwin.log'))

    # Reference linac
    ref_linac = acc.Accelerator(FILEPATH, PROJECT_FOLDER, "Working")
    results = ref_linac.elts.compute_transfer_matrices()
    ref_linac.store_results(results, ref_linac.elts)

    linacs = [ref_linac]

    # Broken linac
    # lsq_info = None

    l_failed = [failed_9]
    l_wtf = [wtf_9]

    # l_failed = [failed_1, failed_2, failed_3, failed_4, failed_5, failed_6,
    #             failed_7]
    # l_wtf = [wtf_1, wtf_2, wtf_3, wtf_4, wtf_5, wtf_6, wtf_7]

    # l_failed = [[all_cavs[i]] for i in range(0, 40)]
    # l_wtf = [wtf_classic for i in range(0, 40)]

    lw_fit_evals = []

# =============================================================================
# Run all simulations of the Project
# =============================================================================
    for [wtf, failed] in zip(l_wtf, l_failed):
        start_time = time.monotonic()
        lin = acc.Accelerator(FILEPATH, PROJECT_FOLDER, "Broken")
        fail = mod_fs.FaultScenario(ref_linac, lin, failed, wtf=wtf)
        linacs.append(deepcopy(lin))

        if FLAG_FIX:
            fail.fix_all()
            results = lin.elts.compute_transfer_matrices()
            lin.store_results(results, lin.elts)

        linacs.append(lin)

        # Output some info onthe quality of the fit
        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time: {delta_t}")

        # Update the .dat filecontent
        tw.update_dat_with_fixed_cavities(
            lin.get('dat_filecontent', to_numpy=False), lin.elts,
            lin.get('field_map_folder'))

        # Reproduce TW's Data tab
        data = tw.output_data_in_tw_fashion(lin)

        # Some measurables to evaluate how the fitting went
        lw_fit_eval = fail.evaluate_fit_quality(delta_t)
        helper.printd(lw_fit_eval, header='Fit evaluation')

        if SAVE_FIX:
            lin.files['dat_filepath'] = os.path.join(
                lin.get('out_lw'), os.path.basename(FILEPATH))

            # Save .dat file, plus other data that is given
            output.save_files(lin, data=data, lw_fit_eval=lw_fit_eval)

        lw_fit_evals.append(lw_fit_eval)


# =============================================================================
# TraceWin
# =============================================================================
    l_fred = []
    l_bruce = []
    if FLAG_TW:
        for lin in linacs:
            # It would be a loss of time to do these simulation
            if 'Broken' in lin.name:
                continue

            # FIXME to modify simulation flags, go to
            # Accelerator.simulate_in_tracewin
            ini_path = FILEPATH.replace('.dat', '.ini')
            lin.simulate_in_tracewin(ini_path, **kwargs_tw)
            lin.store_tracewin_results()

            if 'Fixed' in lin.name:
                lin.resample_tw_results(linacs[0])

            lin.precompute_some_tracewin_results()

            if FLAG_EVALUATE and 'Fixed' in lin.name:
                d_fred = evaluate.fred_tests(linacs[0], lin)
                l_fred.append(d_fred)

                d_bruce = evaluate.bruce_tests(linacs[0], lin)
                l_bruce.append(d_bruce)

        if FLAG_EVALUATE:
            for _list, name in zip([l_fred, l_bruce],
                                   ['fred_tests.csv', 'bruce_tests.csv']):
                out = pd.DataFrame(_list)
                filepath = os.path.join(PROJECT_FOLDER, name)
                out.to_csv(filepath)

# =============================================================================
# Plot
# =============================================================================
    kwargs = {'plot_tw': FLAG_TW, 'save_fig': SAVE_FIX, 'clean_fig': True}
    for i in range(len(l_wtf)):
        for str_plot in PLOTS:
            # Plot the reference linac, i-th broken linac and corresponding
            # fixed linac
            args = (linacs[0], linacs[2 * i + 1], linacs[2 * i + 2])
            plot.plot_preset(str_plot, *args, **kwargs)
