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
ref_values = [2209.9866910478313, 601.1642457863487,
              np.array(([-0.43830142903080804, 0.6169423012578288],
                        [-0.08385571427531216, -0.21305364334117177]))]
print(
    f"delta_phi: {linac.synch.phi['abs_array'][-1]-ref_values[0]}\t",
    f"delta_W: {linac.synch.energy['kin_array_mev'][-1]-ref_values[1]}\t",
    f"delta_MT: {np.sum(linac.transf_mat['cumul'][-1]-ref_values[2])}"
)
