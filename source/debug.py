#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9

font = {'family': 'serif',
        'size':   25}
plt.rc('font', **font)


def plot_error_on_transfer_matrices_components(LINAC):
    """
    Estimate the error on transfer matrix calculation.

    Compare transfer matrices with the one calculated by TraceWin.

    Parameters
    ----------
    LINAC: accelerator object.
        Accelerator under study.
    """
    err = np.full((2, 2, 39), np.NaN)
    idx_min = 0
    for idx_max in range(1, 39):
        R_zz_tot = LINAC.compute_transfer_matrix_and_gamma(idx_min=idx_min,
                                                           idx_max=idx_max)
        err[:, :, idx_max] = compare_to_TW(R_zz_tot, idx_min=idx_min,
                                           idx_max=idx_max)

    fig = plt.figure(20)
    ax = fig.add_subplot(111)
    elt_array = np.linspace(1, 39, 39, dtype=int)
    ax.plot(elt_array, err[0, 0, :], label=r'$R_{11}$')
    ax.plot(elt_array, err[0, 1, :], label=r'$R_{12}$')
    ax.plot(elt_array, err[1, 0, :], label=r'$R_{21}$')
    ax.plot(elt_array, err[1, 1, :], label=r'$R_{22}$')
    ax.legend()
    ax.grid(True)
    ax.set_xlabel('Element #')
    ax.set_ylabel('Error [%]')


def compare_to_TW(R_zz, idx_min, idx_max):
    """
    Compare the transfer matrix calculated by LightWin with TraceWin.

    Parameters
    ----------
    R_zz: np.array
        Transfer matrix between elements idx_min and idx_max.
    idx_min: integer
        Min index.
    idx_max: integer
        index.
    """
    filepath = '/home/placais/LightWin/data/matrix_ref.txt'
    flag_output_errors = False
    R_zz_ref = np.eye(2)
    i = 0

    with open(filepath) as file:
        for line in file:
            elt_number = i // 8
            if(elt_number < idx_min):
                i += 1
                continue
            elif(elt_number == idx_max):
                break
            else:
                if(i % 8 == 6):
                    line1 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]

                elif(i % 8 == 7):
                    line2 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]
                    R_zz_elt = np.vstack((line1, line2))
                    R_zz_ref = np.matmul(R_zz_ref, R_zz_elt)
                i += 1

    R_zz_comp = np.divide((R_zz_ref - R_zz), R_zz_ref) * 100.

    if(flag_output_errors):
        print('TraceWin R_zz:\n', R_zz_ref)
        print(' ')
        print('LightWin R_zz:\n', R_zz)
        print(' ')
        print('Comparison:\n', R_zz_comp)
        print('==============================================================')
        print(' ')
    return R_zz_comp
