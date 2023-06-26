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
from optimisation.fault_scenario import FaultScenario
import tracewin.interface
from beam_calculation.factory import create_beam_calculator_object
from util import output, evaluate
from visualization import plot


if __name__ == '__main__':
    MY_CONFIG_FILE = 'myrrha.ini'
    MY_KEYS = {
        'files': 'files',
        'beam_calculator': 'beam_calculator.lightwin.envelope_longitudinal',
        'beam': 'beam',
        'wtf': 'wtf.k_out_of_n',
        'beam_calculator_post': 'beam_calculator_post.tracewin.quick_debug',
    }

    # =========================================================================
    # Fault compensation
    # =========================================================================
    FLAG_BREAK = True
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
        # "cav",
        "emittance",
        # "twiss",  # TODO
        # "envelopes", # FIXME
        # "transfer matrices", # TODO
    ]

    # =========================================================================
    # Start
    # =========================================================================
    my_configs = conf_man.process_config(MY_CONFIG_FILE, MY_KEYS)

    FILEPATH = my_configs['files']['dat_file']
    PROJECT_FOLDER = my_configs['files']['project_folder']

    ref_linac = acc.Accelerator('Working', **my_configs['files'])
    beam_calculator = \
        create_beam_calculator_object(my_configs['beam_calculator'])
    beam_calculator._init_solver_parameters(ref_linac.elts)

    simulation_output = beam_calculator.run(ref_linac.elts)
    ref_linac.keep_this(simulation_output)

    data_tab_from_tw = tracewin.interface.output_data_in_tw_fashion(ref_linac)
    linacs = [ref_linac]

    lw_fit_evals = []

# =============================================================================
# Run all simulations of the Project
# =============================================================================
    l_failed = my_configs['wtf'].pop('failed')
    l_manual = None
    manual = None
    if 'manual list' in my_configs['wtf']:
        l_manual = my_configs['wtf'].pop('manual list')

    if FLAG_BREAK:
        for i, failed in enumerate(l_failed):
            start_time = time.monotonic()
            lin = acc.Accelerator('Broken', **my_configs['files'])
            beam_calculator._init_solver_parameters(lin.elts)

            if l_manual is not None:
                manual = l_manual[i]
            fault_scenario = FaultScenario(ref_acc=ref_linac,
                                           fix_acc=lin,
                                           beam_calculator=beam_calculator,
                                           wtf=my_configs['wtf'],
                                           fault_idx=failed,
                                           comp_idx=manual)
            linacs.append(deepcopy(lin))

            if FLAG_FIX:
                fault_scenario.fix_all()

            linacs.append(lin)

            end_time = time.monotonic()
            delta_t = datetime.timedelta(seconds=end_time - start_time)
            logging.info(f"Elapsed time: {delta_t}")

            tracewin.interface.update_dat_with_fixed_cavities(
                lin.get('dat_filecontent', to_numpy=False), lin.elts,
                lin.get('field_map_folder'))
            data_tab_from_tw = \
                tracewin.interface.output_data_in_tw_fashion(lin)
            lw_fit_eval = fault_scenario.evaluate_fit_quality(delta_t)

            if SAVE_FIX:
                lin.files['dat_filepath'] = os.path.join(
                    lin.get('out_lw'), os.path.basename(FILEPATH))
                output.save_files(lin, data=data_tab_from_tw,
                                  lw_fit_eval=lw_fit_eval)

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
                lin.files["out_tw"] = os.path.join(os.path.dirname(FILEPATH),
                                                   'ref')
                logging.info(
                    "we do not TW recompute reference linac. "
                    + f"We take TW results from {lin.files['out_tw']}.")
                continue

            ini_path = FILEPATH.replace('.dat', '.ini')
            # TODO transfer ini path elsewhere
            beam_calculator_post = create_beam_calculator_object(
                my_configs['beam_calculator_post'])
            simulation_output = beam_calculator_post.run()

            # tw_simu = TraceWinBeamCalculator(post_tw['executable'],
            #                                  ini_path,
            #                                  lin.get('out_tw'),
            #                                  lin.get('dat_filepath'),
            #                                  post_tw)

            if 'Fixed' in lin.name:
                tracewin.interface.resample_tracewin_results(
                    ref=linacs[0].tracewin_simulation,
                    fix=lin.tracewin_simulation)

            if FLAG_EVALUATE and 'Fixed' in lin.name:
                d_fred = evaluate.fred_tests(linacs[0], lin)
                l_fred.append(d_fred)

                d_bruce = evaluate.bruce_tests(linacs[0], lin)
                l_bruce.append(d_bruce)

        if FLAG_FIX and FLAG_EVALUATE:
            for _list, name in zip([l_fred, l_bruce],
                                   ['fred_tests.csv', 'bruce_tests.csv']):
                out = pd.DataFrame(_list)
                filepath = os.path.join(PROJECT_FOLDER, name)
                out.to_csv(filepath)

# =============================================================================
# Plot
# =============================================================================
    kwargs = {'plot_tw': FLAG_TW, 'save_fig': False, 'clean_fig': True}
    for i in range(len(l_failed)):
        for str_plot in PLOTS:
            # Plot the reference linac, i-th broken linac and corresponding
            # fixed linac
            if not FLAG_BREAK:
                args = (linacs[0], )
            else:
                if not FLAG_FIX:
                    args = (linacs[0], linacs[i + 1])
                else:
                    args = (linacs[0], linacs[2 * i + 1], linacs[2 * i + 2])
            plot.plot_preset(str_plot, *args, **kwargs)
