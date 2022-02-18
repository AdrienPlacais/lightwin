#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""
import os.path
from os import listdir
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Set1_9
from cycler import cycler
import helper
import particle as particle_mod
import transport
from constants import E_rest_MeV
import tracewin_interface as tw

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
    accelerator: Accelerator object.
        Accelerator under study.
    transfer_matrix: numpy array
        Transfer matrices to plot.
    """
    filepath_ref = [accelerator.project_folder + '/results/M_55_ref.txt',
                    accelerator.project_folder + '/results/M_56_ref.txt',
                    accelerator.project_folder + '/results/M_65_ref.txt',
                    accelerator.project_folder + '/results/M_66_ref.txt']

    z_pos = accelerator.synch.z['abs_array']
    n_z = z_pos.shape[0]

    transfer_matrix = accelerator.transf_mat['cumul']

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


def compare_with_tracewin(accelerator, prop, filepath_ref=None):
    """
    Comparison of beam energy with TW data.

    Parameters
    ----------
    accelerator: Accelerator object
        Accelerator under study.
    """
    # Prep
    if filepath_ref is None:
        filepath_ref = accelerator.project_folder + '/results/energy_ref.txt'

    dict_data = {
        'energy': accelerator.synch.energy['kin_array_mev'],
        'abs_phase': np.rad2deg(accelerator.synch.phi['abs_array']),
        'beta_synch': accelerator.synch.energy['beta_array'],
        }
    # Label of property plot, of error plot, figure number
    dict_label = {
        'energy': ['Beam energy [MeV]', 'Abs. error [eV]', 21],
        'abs_phase': ['Beam phase [deg]', 'Abs. error [deg]', 22],
        'beta_synch': [r'Synch. $\beta$ [1]', 'Abs. error [1]', 23],
        }
    # Abs error plot is multiplied by this
    dict_err_factor = {
        'energy': 1e6,
        'abs_phase': 1.,
        'beta_synch': 1.,
        }

    # x axis data
    elt_array = range(accelerator.n_elements)

    # y axis data
    data_ref = tw.load_tw_results(filepath_ref, prop)
    data = []
    for elt in accelerator.list_of_elements:
        idx = elt.idx['out']
        data.append(dict_data[prop][idx])
    data = np.array(data)
    error = np.abs(data_ref - data)

    # Plot
    fig, axlist = helper.create_fig_if_not_exist(dict_label[prop][2],
                                                 range(311, 314))
    if 'TW' not in axlist[0].get_legend_handles_labels()[1]:
        axlist[0].plot(elt_array, data_ref, label='TW', c='k', ls='--')

    axlist[0].plot(elt_array, data, label='LW ' + accelerator.name)
    axlist[1].plot(elt_array, error * dict_err_factor[prop])
    helper.plot_structure(accelerator, axlist[2], x_axis='index')
    axlist[2].set_xlim(axlist[0].get_xlim())

    for i in range(2):
        axlist[i].grid(True)
        axlist[i].set_ylabel(dict_label[prop][i])
    axlist[2].set_xlabel('Element #')
    axlist[0].legend()


def plot_vcav_and_phis(accelerator):
    """
    Plot the evolution of the cavities parameters with s.

    Parameters
    ----------
    accelerator: Accelerator object
        Accelerator under study.
    """
    v_cav_mv = []
    phi_s_deg = []
    idx = []
    i = 0
    for elt in accelerator.list_of_elements:
        if elt.name == 'FIELD_MAP':
            v_cav_mv.append(elt.acc_field.v_cav_mv)
            phi_s_deg.append(elt.acc_field.phi_s_deg)
            idx.append(i)
        i += 1
    fig, ax = helper.create_fig_if_not_exist(25, [311, 312, 313])
    ax[0].plot(idx, v_cav_mv, label='LW ' + accelerator.name, marker='o')
    ax[0].set_ylabel('Acc. voltage [MV]')
    ax[1].plot(idx, phi_s_deg, marker='o')
    ax[1].set_ylabel('Synch. phase [deg]')
    for axx in ax[0:-1]:
        axx.grid(True)

    helper.plot_structure(accelerator, ax[2], x_axis='index')
    ax[2].set_xlim(ax[0].get_xlim())
    ax[2].set_xlabel('Element #')
    ax[0].legend()


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
    # x_axis = 'z'
    x_axis = 'phase'
    # y_axis = 'E'
    y_axis = 'dp/p'
    # y_axis = "z'"

    # Create plot
    fig, ax = helper.create_fig_if_not_exist(41, [111])
    ax = ax[0]
    ax.set_xlabel(r'$\delta z$ [mm]')

    # Set proper y axis and access to proper y data
    if x_axis == 'z':
        ax.set_xlabel(r'$\delta z$ [mm]')
        x_data = {
            'tw': lambda element, i: element['z(mm)'][i],
            'lw': lambda part: part.phase_space['z_array'] * 1e3,
                }

    elif x_axis == 'phase':
        ax.set_xlabel(r'$\phi$ [deg]')
        x_data = {
            'tw': lambda element, i: element['Phase(deg)'][i],
            'lw': lambda part: np.rad2deg(part.phase_space['phi_array_rad']),
                }

    else:
        raise IOError('Wrong x_axis argument in compare_phase_space.')

    # Set proper y axis and access to proper y data
    if y_axis == 'E':
        ax.set_ylabel(r'$E$ [MeV]')
        y_data = {
            'tw': lambda element, i: element['Energy(MeV)'][i],
            'lw': lambda part: part.energy['kin_array_mev'],
                }

    elif y_axis == 'dp/p':
        ax.set_ylabel(r'$dp/p$ [%]')
        y_data = {
            'tw': lambda element, i: helper.mrad_and_mev_to_delta(
                element["z'(mrad)"][i], element['Energy(MeV)'][i], E_rest_MeV),
            'lw': lambda part: part.phase_space['delta_array'] * 100.,
                }

    elif y_axis == "z'":
        ax.set_ylabel(r"$z'$ [mrad]")
        y_data = {
            'tw': lambda element, i: element["z'(mrad)"][i],
            'lw': lambda part: part.phase_space['delta_array']
            * part.energy['gamma_array']**-2 * 1e3,
                }

    else:
        raise IOError('Wrong y_axis argument in compare_phase_space.')

    ax.grid(True)

    # Load TW data
    partran_data = load_phase_space(accelerator)
    n_part = partran_data[0]['x(mm)'].size

    # Plot TW data
    for element in partran_data:
        for i in range(n_part):
            if i in idx_of_part_to_plot:
                ax.scatter(x_data['tw'](element, i), y_data['tw'](element, i),
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
            helper.plot_pty_with_data_tags(ax, x_data['lw'](part),
                                           y_data['lw'](part), idx, tags=True)
        i += 1

    # TODO: remove when useless
    accelerator.particle_list = particle_list
    accelerator.partran_data = partran_data
