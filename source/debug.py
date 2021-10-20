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
import os.path
from tkinter import Tk
from tkinter.filedialog import askopenfilename

font = {'family': 'serif',
        'size':   25}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_9.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def plot_error_on_transfer_matrices_components(filepath_dat, accelerator):
    """
    Estimate the error on transfer matrix calculation.

    Compare transfer matrices with the one calculated by TraceWin.

    Parameters
    ----------
    filepath_dat: str
        Path to the .dat file. The file containing the transfer matrices
        exported by TraceWin is expected to be
        /project_folder/results/matrix_ref.txt, the .dat beeing in
        /project_folder/.
    accelerator: Accelerator object.
        Accelerator under study.
    """
    filepath_ref = '/'.join(filepath_dat.split('/')[:-1])
    filepath_ref = filepath_ref + '/results/matrix_ref.txt'
    if(not os.path.isfile(filepath_ref)):
        print('debug/plot_error_on_transfer_matrices_components error:')
        print('The filepath to the transfer matrices file is invalid. Please')
        print('check the source code for more info. Enter a valid filepath:')
        Tk().withdraw()
        filepath_ref = askopenfilename(
            filetypes=[("TraceWin transfer matrices file", ".txt")])

    n_elts = accelerator.n_elements
    # In this array we store the errors of individual elements
    err_single = np.full((2, 2, n_elts), np.NaN)
    # Here we store the error of the line
    err_tot = np.full((2, 2, n_elts), np.NaN)
    R_zz_tot_ref = np.eye(2)

    # FIXME DIAG element are considered as elements with 0 length, which
    # completely mess the indices and comparisons
    for i in range(n_elts):
        R_zz_single = accelerator.R_zz_single[:, :, i]
        R_zz_tot = accelerator.R_zz_tot_list[:, :, i]

        R_zz_single_ref = import_transfer_matrix_single(filepath_ref, i)
        R_zz_tot_ref = R_zz_single_ref @ R_zz_tot_ref

        err_single[:, :, i] = np.abs(R_zz_single_ref - R_zz_single)
        err_tot[:, :, i] = np.abs(R_zz_tot_ref - R_zz_tot)

        if(i == n_elts - 1):
            print('Single LW: \n', R_zz_single)
            print('Single TW: \n', R_zz_single_ref)
            print('Single err: \n', err_single[:, :, i])
            print('Tot LW: \n', R_zz_tot)
            print('Tot TW: \n', R_zz_tot_ref)
            print('Tot err: \n', err_tot[:, :, i])

    if(plt.fignum_exists(20)):
        fig = plt.figure(20)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
    else:
        fig = plt.figure(20)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

    elt_array = np.linspace(1, n_elts, n_elts, dtype=int)
    ax1.plot(elt_array, err_single[0, 0, :], label=r'$R_{11}$')
    ax1.plot(elt_array, err_single[0, 1, :], label=r'$R_{12}$')
    ax1.plot(elt_array, err_single[1, 0, :], label=r'$R_{21}$')
    ax1.plot(elt_array, err_single[1, 1, :], label=r'$R_{22}$')

    ax2.plot(elt_array, err_tot[0, 0, :])
    ax2.plot(elt_array, err_tot[0, 1, :])
    ax2.plot(elt_array, err_tot[1, 0, :])
    ax2.plot(elt_array, err_tot[1, 1, :])

    ax1.legend()
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_ylabel('Error on single element')
    ax2.set_ylabel('Error from line start')
    ax2.set_xlabel('Element #')


def import_transfer_matrix_single(filepath_ref, idx_element):
    """
    Import the i-th element transfer matrix.

    Parameters
    ----------
    filepath_ref: str
        Filepath to the matrix_ref.txt file.
    idx_element: integer
        Index of the desired transfer matrix.
    """
    flag_output = False
    i = 0
    R_zz_single_ref = np.full((2, 2), np.NaN)

    with open(filepath_ref) as file:
        for line in file:
            elt_number = i // 8

            if(elt_number < idx_element):
                i += 1
                continue
            elif(elt_number > idx_element):
                break
            else:
                if(i % 8 == 6):
                    line1 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]

                elif(i % 8 == 7):
                    line2 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]
                    R_zz_single_ref = np.vstack((line1, line2))
            i += 1

    if(flag_output):
        print('TraceWin R_zz:\n', R_zz_single_ref)
        print(' ', i)
        print('==============================================================')
        print(' ')
    return R_zz_single_ref


def compare_energies(filepath_dat, accelerator):
    """
    Comparison of beam energy with TW data.

    Parameters
    ----------
    filepath_dat: str
        Path to the .dat file. The file containing the energies
        exported by TraceWin ('Save table to file' button in 'Data' tab) is
        expected to be /project_folder/results/energy_ref.txt, the .dat beeing
        in /project_folder/.
    accelerator: Accelerator object
        Accelerator under study.
    """
    flag_output_field_map_acceleration = False
    filepath_ref = '/'.join(filepath_dat.split('/')[:-1])
    filepath_ref = filepath_ref + '/results/energy_ref.txt'
    if(not os.path.isfile(filepath_ref)):
        print('debug/compare_energies error:')
        print('The filepath to the energy file is invalid. Please check the')
        print('source code for more info. Enter a valid filepath:')
        Tk().withdraw()
        filepath_ref = askopenfilename(
            filetypes=[("TraceWin energies file", ".txt")])

    elt_array = np.linspace(1, 39, 39, dtype=int)
    E_MeV_ref = np.full((39), np.NaN)

    i = 0
    with open(filepath_ref) as file:
        for line in file:
            try:
                current_element = line.split('\t')[0]
                current_element = int(current_element)
            except ValueError:
                continue
            splitted_line = line.split('\t')

            # Concerns field maps only:
            if(flag_output_field_map_acceleration and
               accelerator.elements_nature[i] == 'FIELD_MAP'):
                EoTLc_ref = splitted_line[6]
                EoTLc = accelerator.V_cav_MV[i]
                Sych_Phase_ref = splitted_line[8]
                Sync_Phase = accelerator.phi_s_deg[i]
                print('=====================================================')
                print('FIELD_MAP #', i)
                print('V_cav: ', EoTLc, 'MV')
                print('V_cav_ref: ', EoTLc_ref, 'MV')
                err = 1e3 * np.abs(EoTLc - float(EoTLc_ref))
                print('Error: ', err, 'kV')
                print('')
                print('phi_s: ', Sync_Phase, 'deg')
                print('phi_s_deg: ', Sych_Phase_ref, 'deg')
                err = 1e3 * np.abs(Sync_Phase - float(Sych_Phase_ref))
                print('Error: ', err, 'mdeg')
                print('')

            E_MeV_ref[i] = splitted_line[9]
            i += 1

    error = np.abs(E_MeV_ref - accelerator.E_MeV[1:])

    if(plt.fignum_exists(21)):
        fig = plt.figure(21)
        ax1 = fig.axes[0]
        ax2 = fig.axes[1]
    else:
        fig = plt.figure(21)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    ax1.plot(elt_array, accelerator.E_MeV[1:], label='LightWin')
    ax1.plot(elt_array, E_MeV_ref, label='TraceWin')
    ax2.plot(elt_array, error*1e3)
    ax1.grid(True)
    ax2.grid(True)
    ax2.set_xlabel('Element #')
    ax1.set_ylabel('Beam energy [MeV]')
    ax2.set_ylabel('Absolute error [keV]')

    ax1.legend()
