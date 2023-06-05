#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 13:50:45 2022.

@author: placais
"""
import os
import logging
import time
from datetime import timedelta
import numpy as np
import pandas as pd
import accelerator as acc
import fault_scenario as mod_fs
import helper
import matplotlib.pyplot as plt

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
l_failed_cav = [
    # [35],
    # [155, 157],
    [205],
]
what_to_fit = {
    'opti method': 'PSO',
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
    ['energy', 'phase', 'twiss_beta', 'twiss_gamma'],
    ['energy', 'phase', 'eps', 'twiss_beta', 'twiss_gamma'],
    ['energy', 'phase', 'M_11', 'M_12', 'M_22'],
    ['energy', 'phase', 'M_11', 'M_12', 'M_21', 'M_22'],
    ['energy', 'phase', 'mismatch_factor'],
]

measurables = ['Time [s]', r'$W_{kin}$', r'$\phi$', r'$\sigma_\phi$',
               r'$\sigma_W$', '$M$']
df_rank = pd.DataFrame(columns=measurables)

xticks = np.linspace(0, len(measurables), len(measurables))

j = 0
for failed_cav in l_failed_cav:
    fig = plt.figure(30 + j)
    axx = fig.add_subplot(111)
    axx.set_xticks(xticks)
    axx.set_xticklabels(measurables)
    axx.set_yscale('log')
    i = 0
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
        df_rank.loc[i] = fail.evaluate_fit_quality(delta_t.seconds)
        axx.plot(xticks, df_rank.loc[i], label=str(i))
        i += 1

    axx.grid(True)
    axx.legend()
    j += 1

logging.info(helper.pd_output(df_rank, header='bonjoure'))
