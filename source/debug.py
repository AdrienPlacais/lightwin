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
import helper
from scipy.interpolate import interp1d

font = {'family': 'serif',
        'size':   20}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_9.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def compute_error_transfer_matrix(transf_mat, transf_mat_ref,
                                  flag_output=False):
    """Compute and output error between transfer matrix and ref."""
    n_z = transf_mat.shape[0]
    n_z_ref = transf_mat_ref.shape[0]

    # We calculate error by interpolating the tab with most points on the one
    # with least points.
    kind = 'linear'
    bounds_error = False
    fill_value = 'extrapolate'

    if n_z < n_z_ref:
        z_err = transf_mat[:, 0]
        err = np.full((n_z, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=transf_mat_ref[:, 0],
                                y=transf_mat_ref[:, i+1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = f_interp(z_err) - transf_mat[:, i+1]

    else:
        z_err = transf_mat_ref[:, 0]
        err = np.full((n_z_ref, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=transf_mat[:, 0],
                                y=transf_mat[:, i+1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = transf_mat_ref[:, i+1] - f_interp(z_err)

    if flag_output:
        print('=============================================================')
        print('Error matrix at end of line*1e3:\n',
              err[-1, 0:2]*1e3, '\n',
              err[-1, 2:4]*1e3)
        print('')
        print('Cumulated error:\n',
              np.linalg.norm(err, axis=0)[0:2], '\n',
              np.linalg.norm(err, axis=0)[2:4])
        print('')
        print('Tot error:\n', np.linalg.norm(err))
        print('=============================================================')
    return err, z_err


def load_transfer_matrices(filepath_list):
    """Load transfer matrices saved in 4 files by components."""
    i = 0
    for path in filepath_list:
        assert os.path.isfile(path), \
            'Incorrect filepath in plot_transfer_matrices.'

        if i == 0:
            r_zz_tot_ref = np.loadtxt(filepath_list[i])

        else:
            tmp = np.loadtxt(filepath_list[i])[:, 1]
            tmp = np.expand_dims(tmp, axis=1)
            r_zz_tot_ref = np.hstack((r_zz_tot_ref, tmp))
        i += 1

    return r_zz_tot_ref


def plot_transfer_matrices(accelerator):
    """
    Plot the transfer matrix components of TraceWin and LightWin.

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
    filepath_ref = [accelerator.project_folder + '/results/M_55_ref.txt',
                    accelerator.project_folder + '/results/M_56_ref.txt',
                    accelerator.project_folder + '/results/M_65_ref.txt',
                    accelerator.project_folder + '/results/M_66_ref.txt']

    z = accelerator.get_from_elements('pos_m')
    n_z = z.shape[0]

    # Change shape of calculated transfer matrix to match the ref one
    # i.e.: 1st column is z, 2nd 3rd 4th and 5th are matrix components
    r_zz_tot = accelerator.transfer_matrix_cumul.reshape((n_z, 4))
    r_zz_tot = np.hstack((np.expand_dims(z, 1), r_zz_tot))

    r_zz_tot_ref = load_transfer_matrices(filepath_ref)

    err, z_err = compute_error_transfer_matrix(r_zz_tot, r_zz_tot_ref, True)

    fignum = 26
    fig, axlist = helper.create_fig_if_not_exist(fignum, range(221, 225))

    if helper.empty_fig(fignum):
        ls = '-'
        labels_tw = ['TraceWin', '', '', '']
        for i in range(4):
            axlist[i].plot(r_zz_tot_ref[:, 0], r_zz_tot_ref[:, i+1],
                           label=labels_tw[i], ls=ls)
    else:
        ls = '--'

    xlabels = ['', '', 'z [m]', 'z [m]']
    ylabels = [r'$R_{11}$', r'$R_{12}$', r'$R_{21}$', r'$R_{22}$']
    labels_lw = ['LightWin', '', '', '']

    for i in range(4):
        axlist[i].plot(r_zz_tot[:, 0], r_zz_tot[:, i+1],
                       label=labels_lw[i], ls=ls)
        axlist[i].set_xlabel(xlabels[i])
        axlist[i].set_ylabel(ylabels[i])
        axlist[i].grid(True)

    axlist[0].legend()

    axlist = []
    fignum *= 10
    fig, axlist = helper.create_fig_if_not_exist(fignum, range(221, 225))
    if helper.empty_fig(fignum):
        ls = '-'
    else:
        ls = '--'

    xlabels = ['', '', 'z [m]', 'z [m]']
    ylabels = [r'$\epsilon R_{11}$', r'$\epsilon R_{12}$',
               r'$\epsilon R_{21}$', r'$\epsilon R_{22}$']

    for i in range(4):
        axlist[i].plot(z_err, err[:, i], ls=ls)
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
    r_zz_single_ref = np.full((2, 2), np.NaN)

    with open(filepath_ref) as file:
        for line in file:
            elt_number = i // 8

            if elt_number < idx_element:
                i += 1
                continue
            elif elt_number > idx_element:
                break
            else:
                if i % 8 == 6:
                    line1 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]

                elif i % 8 == 7:
                    line2 = np.fromstring(line, dtype=float,
                                          count=6, sep=' ')[-2:]
                    r_zz_single_ref = np.vstack((line1, line2))
            i += 1

    if flag_output:
        print('TraceWin r_zz:\n', r_zz_single_ref)
        print(' ', i)
        print('==============================================================')
        print(' ')
    return r_zz_single_ref


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
    if not os.path.isfile(filepath_ref):
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

    # error = np.abs(E_MeV_ref - accelerator.E_MeV[1:])
    calculated_energy = np.full((accelerator.n_elements), np.NaN)
    for i in range(accelerator.n_elements):
        calculated_energy[i] = \
            accelerator.list_of_elements[i].energy_array_mev[-1]

    error = np.abs(E_MeV_ref - calculated_energy)

    if plt.fignum_exists(21):
        fig = plt.figure(21)
        axlist = [fig.axes[0], fig.axes[1]]

    else:
        fig = plt.figure(21)
        axlist = [fig.add_subplot(211), fig.add_subplot(212)]

    if helper.empty_fig(21):
        axlist[0].plot(elt_array, E_MeV_ref, label='TraceWin')

    axlist[0].plot(elt_array, calculated_energy, label='LightWin')
    axlist[1].plot(elt_array, error*1e6)

    for ax in axlist:
        ax.grid(True)

    axlist[0].set_ylabel('Beam energy [MeV]')
    axlist[1].set_xlabel('Element #')
    axlist[1].set_ylabel('Absolute error [eV]')

    axlist[0].legend()
