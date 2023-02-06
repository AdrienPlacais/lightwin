#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:33:39 2022.

@author: placais
"""

import os
from copy import deepcopy
import time
import datetime
import core.accelerator as acc
import core.fault_scenario as mod_fs
from util import debug, helper, output
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

    failed_0 = [12]
    wtf_0 = {'opti method': 'least_squares', 'strategy': 'k out of n',
             'k': 5, 'l': 2, 'manual list': [],
             'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
             'position': 'end_mod', 'phi_s fit': True}

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
    PROJECT_FOLDER = os.path.join(
        os.path.dirname(FILEPATH),
        datetime.datetime.now().strftime('%Y.%m.%d_%Hh%M_%Ss_%fms'))

    # Reference linac
    ref_linac = acc.Accelerator(FILEPATH, PROJECT_FOLDER, "Working")
    results = ref_linac.elts.compute_transfer_matrices()
    ref_linac.store_results(results, ref_linac.elts)

    linacs = [ref_linac]

    # Broken linac
    # lsq_info = None
    # l_failed = [failed_1, failed_2, failed_3, failed_4, failed_5, failed_6,
    #             failed_7]
    # l_wtf = [wtf_1, wtf_2, wtf_3, wtf_4, wtf_5, wtf_6, wtf_7]
    l_failed = [failed_0]
    l_wtf = [wtf_0]

    for [wtf, failed] in zip(l_wtf, l_failed):
        name = failed[0]
        if isinstance(name, list):
            name = name[0]
        name = str(name)
        start_time = time.monotonic()
        lin = acc.Accelerator(FILEPATH, PROJECT_FOLDER, "Broken " + name)
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
        print(f"\n\nElapsed time: {delta_t}")

        # Update the .dat filecontent
        tw.update_dat_with_fixed_cavities(lin.get('dat_filecontent'), lin.elts,
                                          lin.get('field_map_folder'))
        # Reproduce TW's Data tab
        data = tw.output_data_in_tw_fashion(lin)

        # Some measurables to evaluate how the fitting went
        ranking = fail.evaluate_fit_quality(delta_t)
        helper.printd(ranking, header='Fit evaluation')

        if SAVE_FIX:
            lin.files['dat_filepath'] = os.path.join(
                lin.get('out_lw'), os.path.basename(FILEPATH))

            # Save .dat file, plus other data that is given
            output.save_files(lin, data=data, ranking=ranking)

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
        # project = linac.get('out_tw')
        project = '/home/placais/LightWin/data/JAEA/2023.02.02_16h25_56s_0/TW'
        debug.compare_with_multiparticle_tw(project)
