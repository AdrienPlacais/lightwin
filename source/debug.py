#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_4
from cycler import cycler
import os.path
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import helper
from scipy.interpolate import interp1d

font = {'family': 'serif',
        'size':   25}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_4.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def plot_error_on_transfer_matrices_components_simple(filepath_dat,
                                                      accelerator):
    """
    Estimate the error on transfer matrix calculation.

    Compare transfer matrices with the one calculated by TraceWin. Plot
    as a function of element number.

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
    # completely messes with the indices and comparisons
    for i in range(n_elts):
        R_zz_single = np.copy(accelerator.R_zz_single[:, :, i])
        R_zz_tot = np.copy(accelerator.R_zz_tot_list[:, :, i])

        R_zz_single_ref = import_transfer_matrix_single(filepath_ref, i)
        R_zz_tot_ref = R_zz_single_ref @ R_zz_tot_ref

        err_single[:, :, i] = R_zz_single_ref - R_zz_single
        err_tot[:, :, i] = R_zz_tot_ref - R_zz_tot

        if(i == -1):
            print(' ')
            print('=========================================================')
            print(' ')
            print('LightWin version:')
            print('Single LW: \n', R_zz_single, '\n')
            print('Single TW: \n', R_zz_single_ref, '\n')
            print('Single err*1e3: \n', 1e3*err_single[:, :, i], '\n')
            print(' ')
            print('=========================================================')
            print(' ')
            print('Tot LW: \n', R_zz_tot, '\n')
            print('Tot TW: \n', R_zz_tot_ref, '\n')
            print('Tot err*1e3: \n', 1e3*err_tot[:, :, i], '\n')
            print(' ')
            print('=========================================================')

    if(plt.fignum_exists(20)):
        fig = plt.figure(20)
        axlist = [fig.axes[0], fig.axes[1]]

    else:
        fig = plt.figure(20)
        axlist = [fig.add_subplot(211), fig.add_subplot(212)]

    if(helper.empty_fig(20)):
        ls = '-'
    else:
        ls = '--'

    elt_array = np.linspace(1, n_elts, n_elts, dtype=int)
    labels = [r'$R_{11}$', r'$R_{12}$', r'$R_{21}$', r'$R_{22}$']

    for i in range(4):
        axlist[0].plot(elt_array, err_single[i//2, i % 2, :],
                       label=labels[i], ls=ls)
        axlist[1].plot(elt_array, err_tot[i//2, i % 2, :], ls=ls)

    axlist[0].legend()

    for ax in axlist:
        ax.grid(True)
    axlist[0].set_ylabel('Error on single element')
    axlist[1].set_ylabel('Error from line start')
    axlist[1].set_xlabel('Element #')


def plot_error_on_transfer_matrices_components_full(filepath_dat,
                                                    accelerator):
    """
    Estimate the error on transfer matrix calculation.

    Compare transfer matrices with the one calculated by TraceWin. Plot as a
    function of z.

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
    filepath_ref = [filepath_ref + '/results/M_55_ref.txt',
                    filepath_ref + '/results/M_56_ref.txt',
                    filepath_ref + '/results/M_65_ref.txt',
                    filepath_ref + '/results/M_66_ref.txt']
    R_zz_tot = accelerator.full_MT_and_energy_evolution[:, 1:, :]
    z = accelerator.full_MT_and_energy_evolution[:, 0, 0]

    i = 0
    for path in filepath_ref:
        if(not os.path.isfile(path)):
            print('debug/plot_error_on_transfer_matrices_components error.')

        if(i == 0):
            R_zz_tot_ref = np.loadtxt(filepath_ref[i])
        else:
            tmp = np.loadtxt(filepath_ref[i])[:, 1]
            tmp = np.expand_dims(tmp, axis=1)
            R_zz_tot_ref = np.hstack((R_zz_tot_ref, tmp))
        i += 1

    n_z_ref = R_zz_tot_ref.shape[0]
    n_z = z.shape[0]

    axlist = []
    fignum = 25
    if(plt.fignum_exists(fignum)):
        fig = plt.figure(fignum)
        for i in range(4):
            axlist.append(fig.axes[i])

    else:
        fig = plt.figure(fignum)
        for i in range(221, 225):
            axlist.append(fig.add_subplot(i))

    if(helper.empty_fig(fignum)):
        ls = '-'
    else:
        ls = '--'

    xlabels = ['', '', 'z [m]', 'z [m]']
    ylabels = [r'$R_{11}$', r'$R_{12}$', r'$R_{21}$', r'$R_{22}$']
    labels_TW = ['TW', '', '', '']
    labels_LW = ['LW', '', '', '']

    for i in range(4):
        axlist[i].plot(R_zz_tot_ref[:, 0], R_zz_tot_ref[:, i+1],
                       label=labels_TW[i], ls=ls)
        axlist[i].plot(z, R_zz_tot[:, i // 2, i % 2],
                       label=labels_LW[i], ls=ls)
        axlist[i].set_xlabel(xlabels[i])
        axlist[i].set_ylabel(ylabels[i])
        axlist[i].grid(True)

    axlist[0].legend()

    # We calculate error by interpolating the tab with most points on the one
    # with least points.
    kind = 'linear'
    bounds_error = False
    fill_value = 'extrapolate'

    if(n_z < n_z_ref):
        z_error = z
        err = np.full((n_z, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=R_zz_tot_ref[:, 0],
                                y=R_zz_tot_ref[:, i+1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = f_interp(z_error) - R_zz_tot[:, i // 2, i % 2]

    else:
        z_error = R_zz_tot_ref[:, 0]
        err = np.full((n_z_ref, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=z,
                                y=R_zz_tot[:, i // 2, i % 2],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = R_zz_tot_ref[:, i+1] - f_interp(z_error)

    axlist = []
    fignum *= 10
    if(plt.fignum_exists(fignum)):
        fig = plt.figure(fignum)
        for i in range(4):
            axlist.append(fig.axes[i])

    else:
        fig = plt.figure(fignum)
        for i in range(221, 225):
            axlist.append(fig.add_subplot(i))

    if(helper.empty_fig(fignum)):
        ls = '-'
    else:
        ls = '--'

    xlabels = ['', '', 'z [m]', 'z [m]']
    ylabels = [r'$\epsilon R_{11}$', r'$\epsilon R_{12}$',
               r'$\epsilon R_{21}$', r'$\epsilon R_{22}$']

    for i in range(4):
        axlist[i].plot(z_error, err[:, i], ls=ls)
        axlist[i].set_xlabel(xlabels[i])
        axlist[i].set_ylabel(ylabels[i])
        axlist[i].grid(True)


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
        axlist = [fig.axes[0], fig.axes[1]]

    else:
        fig = plt.figure(21)
        axlist = [fig.add_subplot(211), fig.add_subplot(212)]

    if(helper.empty_fig(21)):
        axlist[0].plot(elt_array, E_MeV_ref, label='TraceWin')

    axlist[0].plot(elt_array, accelerator.E_MeV[1:], label='LightWin')
    axlist[1].plot(elt_array, error*1e6)
    
    for ax in axlist:
        ax.grid(True)

    axlist[0].set_ylabel('Beam energy [MeV]')
    axlist[1].set_xlabel('Element #')
    axlist[1].set_ylabel('Absolute error [eV]')

    axlist[0].legend()
