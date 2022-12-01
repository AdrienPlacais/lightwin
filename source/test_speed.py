#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:37:16 2022.

@author: placais

Run with
%timeit run test_speed.py

Warning! submodules are not always properly reloaded with this command. Hence,
you may want to double check which functions are used, eg after modifying
constants.METHOD.
"""

import os
import numpy as np
import constants
import accelerator as acc
import fault_scenario as mod_fs

# TEST = 'simple'
TEST = 'compensation'

FILEPATH = os.path.abspath(
    "../data/JAEA/JAEA_ADS_026.dat"
    # "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
)

if TEST == 'simple':
    linac = acc.Accelerator(FILEPATH, "Working")
    linac.compute_transfer_matrices()

    # From TW
    ref_values = [np.deg2rad(126622.65), 601.16554,
                  np.array(([-0.43919173, 0.61552512],
                            [-0.083619398, -0.21324773]))]

    print(
        f"delta_phi: {linac.synch.phi['abs_array'][-1]-ref_values[0]}\t"
        + f"delta_W: {linac.synch.energy['kin_array_mev'][-1]-ref_values[1]}\t"
        + f"delta_MT: {np.sum(linac.transf_mat['cumul'][-1]-ref_values[2])}\t"
        + f"METHOD: {constants.METHOD}"
    )

elif TEST == 'compensation':
    ref_linac = acc.Accelerator(FILEPATH, 'Working')
    results = ref_linac.elts.compute_transfer_matrices()
    ref_linac.save_results(results, ref_linac.elts)

    failed_cav = [25]
    wtf_lsq = {'opti method': 'least_squares',
               'strategy': 'k out of n',
               'k': 6, 'l': 2, 'manual list': [],
               'objective': ['w_kin', 'phi_abs_array', 'mismatch factor'],
               'position': 'end_mod', 'phi_s fit': True}
    broken_linac = acc.Accelerator(FILEPATH, "Broken")
    fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav,
                                wtf=wtf_lsq)
    fail.fix_all()
    results = broken_linac.elts.compute_transfer_matrices()
    broken_linac.save_results(results, broken_linac.elts)
