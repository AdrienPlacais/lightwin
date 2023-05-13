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
    FLAG_FIX = True
    SAVE_FIX = True
    FLAG_TW = False
    RECOMPUTE_REFERENCE = False
    FLAG_EVALUATE = False

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
    CONFIG_PATH = 'jaea_default.ini'
    os.makedirs(PROJECT_FOLDER)

    set_up_logging(logfile_file=os.path.join(PROJECT_FOLDER, 'lightwin.log'))

    d_beam, d_flags, wtf_0, d_tw = conf_man.process_config(
        CONFIG_PATH, PROJECT_FOLDER, key_beam='beam.jaea',
        key_wtf='wtf.k_out_of_n', key_tw='tracewin')
    failed_0 = [12]

    # Reference linac
    ref_linac = acc.Accelerator(FILEPATH, PROJECT_FOLDER, "Working")
    results = ref_linac.elts.compute_transfer_matrices()
    ref_linac.store_results(results, ref_linac.elts)

    linacs = [ref_linac]

    l_failed = [failed_0]
    l_wtf = [wtf_0]

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

            if 'Working' in lin.name and not RECOMPUTE_REFERENCE:
                lin.files["out_tw"] = '/home/placais/LightWin/data/JAEA/ref/'
                logging.info(
                    "we do not TW recompute reference linac. "
                    + f"We take TW results from {lin.files['out_tw']}.")
                continue

            ini_path = FILEPATH.replace('.dat', '.ini')
            lin.simulate_in_tracewin(ini_path, **d_tw)
            # TODO transfer ini path elsewhere
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
