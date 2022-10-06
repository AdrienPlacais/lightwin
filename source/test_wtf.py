#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:50:45 2022.

@author: placais
"""
import os
import time
from datetime import timedelta
import accelerator as acc
import fault_scenario as mod_fs

# =============================================================================
# Set linac
# =============================================================================
FILEPATH = os.path.abspath(
    "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat")

ref_linac = acc.Accelerator(FILEPATH, 'Working')
ref_linac.compute_transfer_matrices()

# =============================================================================
# Set the configurations that will be tested
# =============================================================================
l_failed_cav = [[35],
                [205]]
what_to_fit = {
    'opti method': 'least_squares',
    'manual list': [
        [25, 37],
        [145, 147, 165, 175, 177]
    ],
    'strategy': 'l neighboring lattices',
    'k': 2,
    'l': 2,
    'objective': [
        'energy',
        'phase',
        # 'eps', 'twiss_beta', 'twiss_gamma',  # 'twiss_alpha',
        'M_11', 'M_12', 'M_22',  # 'M_21',
        # 'mismatch_factor',
    ],
    'position': 'end_mod',
}
l_objectives = [
    ['energy', 'phase'],
    ['energy', 'phase', 'eps', 'twiss_beta', 'twiss_gamma'],
    ['energy', 'phase', 'M_11', 'M_12', 'M_22'],
    ['energy', 'phase', 'M_11', 'M_12', 'M_21', 'M_22'],
    ['energy', 'phase', 'mismatch_factor'],
]

for failed_cav in l_failed_cav:
    for objective in l_objectives:
        start_time = time.monotonic()
        broken_linac = acc.Accelerator(FILEPATH, "Broken")
        what_to_fit['objective'] = objective
        fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav,
                                    what_to_fit)
        fail.fix_all()
        broken_linac.compute_transfer_matrices()
        end_time = time.monotonic()
        delta_t = timedelta(seconds=end_time - start_time)
        fail.evaluate_fit_quality(delta_t)
