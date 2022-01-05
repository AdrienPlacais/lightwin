#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""
import os.path
from os import listdir
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
from cycler import cycler
import helper
import particle as particle_mod
import transport
from constants import m_MeV

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


def plot_transfer_matrices(accelerator, transfer_matrix):
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

    z_pos = accelerator.synch.z['abs_array']
    n_z = z_pos.shape[0]

    # Change shape of calculated transfer matrix to match the ref one
    # i.e.: 1st column is z, 2nd 3rd 4th and 5th are matrix components
    # r_zz_tot = accelerator.transfer_matrix_cumul.reshape((n_z, 4))
    r_zz_tot = transfer_matrix.reshape((n_z, 4))
    r_zz_tot = np.hstack((np.expand_dims(z_pos, 1), r_zz_tot))

    r_zz_tot_ref = load_transfer_matrices(filepath_ref)

    err, z_err = compute_error_transfer_matrix(r_zz_tot, r_zz_tot_ref, True)

    fignum = 26
    axnumlist = range(221, 225)
    fig, axlist = helper.create_fig_if_not_exist(fignum, axnumlist)

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
    fig, axlist = helper.create_fig_if_not_exist(fignum, axnumlist)
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


def load_energies(filepath, n_elt):
    """Load energy ref file."""
    if not os.path.isfile(filepath):
        print('debug/compare_energies error:')
        print('The filepath to the energy file is invalid. Please check the')
        print('source code for more info. Enter a valid filepath:')
        Tk().withdraw()
        filepath = askopenfilename(
            filetypes=[("TraceWin energies file", ".txt")])

    e_mev_ref = np.full((n_elt), np.NaN)

    i = 0
    with open(filepath) as file:
        for line in file:
            try:
                current_element = line.split('\t')[0]
                current_element = int(current_element)
            except ValueError:
                continue
            splitted_line = line.split('\t')

            e_mev_ref[i] = splitted_line[9]
            i += 1

    return e_mev_ref


def compare_energies(accelerator):
    """
    Comparison of beam energy with TW data.

    Parameters
    ----------
    accelerator: Accelerator object
        Accelerator under study.
    """
    n_elt = accelerator.n_elements
    elt_array = np.linspace(1, n_elt, n_elt, dtype=int)

    filepath_ref = accelerator.project_folder + '/results/energy_ref.txt'
    e_mev_ref = load_energies(filepath_ref, n_elt)

    e_mev = np.full((n_elt), np.NaN)
    # We take energy at the exit of every element
    for i in range(n_elt):
        idx_out = accelerator.list_of_elements[i].idx['out']
        e_mev[i] = accelerator.synch.energy['e_array_mev'][idx_out]

    error = np.abs(e_mev_ref - e_mev)

    fignum = 21
    axnumlist = range(211, 213)
    fig, axlist = helper.create_fig_if_not_exist(fignum, axnumlist)

    if helper.empty_fig(fignum):
        axlist[0].plot(elt_array, e_mev_ref, label='TraceWin')

    axlist[0].plot(elt_array, e_mev, label='LightWin')
    axlist[1].plot(elt_array, error*1e6)

    for ax in axlist:
        ax.grid(True)

    axlist[0].set_ylabel('Beam energy [MeV]')
    axlist[1].set_xlabel('Element #')
    axlist[1].set_ylabel('Absolute error [eV]')
    axlist[0].legend()


def load_phase_space(accelerator):
    """
    Load Partran phase-space data.

    Phase-space files are obtained with:
        Input data & Beam: Partran
        Phase spaces or beam distributions: Output at element n
        Then save all particle as ASCII.
    """
    folder = accelerator.project_folder + '/results/phase_space/'
    file_type = ['txt']
    file_list = []

    for file in listdir(folder):
        if file.split('.')[-1] in file_type:
            file_list.append(folder + file)
    file_list.sort()

    partran_data = []
    dtype = {'names': ('x(mm)', "x'(mrad)", 'y(mm)', "y'(mrad)", 'z(mm)',
                       "z'(mrad)", 'Phase(deg)', 'Time(s)', 'Energy(MeV)',
                       'Loss',),
             'formats': ('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8',
                         'f8', 'i4')}
    for file in file_list:
        partran_data.append(np.loadtxt(file, skiprows=3, dtype=dtype))

    return partran_data


def compare_phase_space(accelerator):
    """
    Compare phase-spaces computed by LightWin and TraceWin (Partran).

    Bonjoure.
    """
    idx_of_part_to_plot = [5, 8]
    # Create plot
    fig, ax = helper.create_fig_if_not_exist(41, [111])
    ax = ax[0]
    ax.set_xlabel(r'$\delta z$ [mm]')
    ax.set_ylabel(r'$dp/p$ [%]')
    ax.grid(True)
    # ax.set_xlim([-2.5, 2.5])
    # ax.set_ylim([-0.1, 0.1])

    # Load TW data
    partran_data = load_phase_space(accelerator)
    n_part = partran_data[0]['x(mm)'].size

    # Plot TW data
    for element in partran_data:
        for i in range(n_part):
            if i in idx_of_part_to_plot:
                dp_p = element["z'(mrad)"][i] \
                    * helper.mev_to_gamma(element['Energy(MeV)'][i],
                                          m_MeV)**2 * 1e-1  # Not 1e-2??
                ax.scatter(element['z(mm)'][i],
                           # element['Energy(MeV)'][i],
                           dp_p,
                           color='k', marker='x')

    # Compute LW data
    particle_list = []
    for i in range(n_part):
        particle_list.append(particle_mod.Particle(
            z=partran_data[0]['z(mm)'][i] * 1e-3,
            e_mev=partran_data[0]['Energy(MeV)'][i],
            omega0_bunch=accelerator.synch.omega0['bunch']))

        transport.transport_particle(accelerator, particle_list[i])
        particle_list[i].compute_phase_space_tot(accelerator.synch)

    # Plot LW data
    idx = accelerator.get_from_elements('idx', 'in')

    i = 0
    for part in particle_list:
        if i in idx_of_part_to_plot:
            helper.plot_pty_with_data_tags(
                ax,
                part.phase_space['z_array'] * 1e3,
                part.phase_space['delta_array'] * 100.,
                # part.energy['e_array_mev'],
                idx, tags=True)
        i += 1
