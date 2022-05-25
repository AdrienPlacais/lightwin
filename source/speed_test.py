#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 15:37:16 2022.

@author: placais

Run with
%timeit run speed_test.py
"""

import os
import numpy as np
import accelerator as acc

FILEPATH = os.path.abspath(
    "../data/faultcomp22/working/MYRRHA_Transi-100MeV.dat"
    # "../data/work_field_map/work_field_map.dat"
)
linac = acc.Accelerator(FILEPATH, "Working")
linac.compute_transfer_matrices()

# From TW
ref_values = [np.deg2rad(126622.65), 601.16554,
              np.array(([-0.43919173, 0.61552512],
                        [-0.083619398, -0.21324773]))]

print(
    f"delta_phi: {linac.synch.phi['abs_array'][-1]-ref_values[0]}\t"
    + f"delta_W: {linac.synch.energy['kin_array_mev'][-1]-ref_values[1]}\t"
    + f"delta_MT: {np.sum(linac.transf_mat['cumul'][-1]-ref_values[2])}\n"
)


# ref_error = [0.004540947433270048, -0.0012942136512492652,
#              0.0022652526105367693]
# print(
#     "Results with _p:\n"
#     + f"delta_phi: {ref_error[0]}\t"
#     + f"delta_W: {ref_error[1]}\t"
#     + f"delta_MT: {ref_error[2]}"
# )
