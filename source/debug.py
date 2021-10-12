#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
from cycler import cycler

font = {'family': 'serif',
        'size':   25}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_9.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def plot_error_on_transfer_matrices_components(LINAC):
    """
    Estimate the error on transfer matrix calculation.

    Compare transfer matrices with the one calculated by TraceWin.

    Parameters
    ----------
    LINAC: accelerator object.
        Accelerator under study.
    """
    n_elts = 6
    err = np.full((2, 2, n_elts), np.NaN)
    idx_min = 0
    i = 0
    for idx_max in range(idx_min + 1, idx_min + 1 + n_elts):
        R_zz_tot = LINAC.compute_transfer_matrix_and_gamma(idx_min=idx_min,
                                                           idx_max=idx_max)
        err[:, :, i] = compare_to_TW(R_zz_tot, idx_min=idx_min,
                                     idx_max=idx_max)
        i += 1

    if(plt.fignum_exists(20)):
        fig = plt.figure(20)
        ax = fig.axes[0]
    else:
        fig = plt.figure(20)
        ax = fig.add_subplot(111)
    elt_array = np.linspace(1, n_elts, n_elts, dtype=int)
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
    filepath = '../data/matrix_ref.txt'
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


def compare_energies(LINAC):
    """
    Comparison of beam energy with TW data.

    Parameters
    ----------
    LINAC: Accelerator object
        Accelerator under study.
    """
    filepath = '../data/energy_ref.txt'
    elt_array = np.linspace(1, 39, 39, dtype=int)
    E_MeV_ref = np.full((39), np.NaN)

    i = 0
    with open(filepath) as file:
        for line in file:
            try:
                current_element = line.split('\t')[0]
                current_element = int(current_element)
            except ValueError:
                continue
            E_MeV_ref[i] = line.split('\t')[9]
            i += 1

    error = (E_MeV_ref - LINAC.E_MeV[1:]) / E_MeV_ref * 100.

    if(plt.fignum_exists(21)):
        fig = plt.figure(21)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
    else:
        fig = plt.figure(21)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    ax1.plot(elt_array, LINAC.E_MeV[1:], label='LightWin')
    ax1.plot(elt_array, E_MeV_ref, label='TraceWin')
    ax2.plot(elt_array, error)
    ax1.grid(True)
    ax2.grid(True)
    ax2.set_xlabel('Element #')
    ax1.set_ylabel('Beam energy [MeV]')
    ax2.set_ylabel('Relative error [%]')

    ax1.legend()

