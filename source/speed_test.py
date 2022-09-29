#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:37:16 2022.

@author: placais

Run with
%timeit run speed_test.py

Warning! submodules are not always properly reloaded with this command. Hence,
you may want to double check which functions are used, eg after modifying
constants.METHOD.
"""

import os
import numpy as np
import constants
import accelerator as acc
import fault_scenario as mod_fs

test = 'simple'
# test = 'compensation'

FILEPATH = os.path.abspath(
    "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
)

if test == 'simple':
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

elif test == 'compensation':
    ref_linac = acc.Accelerator(FILEPATH, 'Working')
    broken_linac = acc.Accelerator(FILEPATH, "Broken")

    failed_cav = [35]
    fail = mod_fs.FaultScenario(ref_linac, broken_linac, failed_cav)

    ref_linac.compute_transfer_matrices()
    fail.transfer_phi0_from_ref_to_broken()
    broken_linac.compute_transfer_matrices()

    fail.prepare_compensating_cavities_of_all_faults([0])
    fail.fix_all()
    broken_linac.compute_transfer_matrices()
