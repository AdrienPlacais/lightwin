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
import pandas as pd
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
    fold = accelerator.files['project_folder']
    filepath_ref = [fold + '/results/M_55_ref.txt',
                    fold + '/results/M_56_ref.txt',
                    fold + '/results/M_65_ref.txt',
                    fold + '/results/M_66_ref.txt']

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


def _reformat(x_data, y_data, elts_indexes):
    """
    Downsample x_data or y_data if it has more points than the other.

    Parameters
    ----------
    x_data : np.array
        Data to plot in x-axis.
    y_data : TYPE
        Data to plot on y-axis.

    Returns
    -------
    x_data : np.array
        Same array, downsampled to elements position if necessary.
    y_data : np.array
        Same array, downsampled to elements position if necessary.
    """
    # Check the shapes
    if x_data.shape[0] < y_data.shape[0]:
        y_data = y_data[elts_indexes]
    elif x_data.shape[0] > y_data.shape[0]:
        x_data = x_data[elts_indexes]
    # Check the NaN values
    valid_idx = np.where(~np.isnan(y_data))
    x_data = x_data[valid_idx]
    y_data = y_data[valid_idx]
    return x_data, y_data


def compare_with_tracewin(linac, x_dat='s',
                          y_dat=['energy', 'energy_err', 'struct'],
                          filepath_ref=None, fignum=21):
    """
    Compare data calculated by TraceWin and LightWin.

    There are several plots on top of each other. Number of plots is determined
    by the len of y_dat.

    Parameters
    ----------
    linac : Accelerator object
        Accelerator under study.
    x_dat : string
        Data in x axis, common to the n plots. It should be 's' for a plot as
        a function of the position and 'elt' for a plot a function of the
        number of elements.
    y_dat : list of string
        Data in y axis for each subplot. It should be in dict_y_data_lw.
    filepath_ref : string
        Path to the TW results. They should be saved in TraceWin: Data > Save
        table to file (loaded by tracewin_interface.load_tw_results).
    fignum: int
        Number of the Figure.
    """
    # [label, marker]
    dict_plot = {
        's': ['Synch. position [m]', '.'],
        'elt': ['Element number',  '.'],
        'energy': ['Beam energy [MeV]',  '.'],
        'energy_err': ['Abs. error [keV]',  '.'],
        'abs_phase': ['Beam phase [deg]',  '.'],
        'abs_phase_err': ['Abs. phase error [deg]',  '.'],
        'beta_synch': [r'Synch. $\beta$ [1]',  '.'],
        'beta_synch_err': [r'Abs. $\beta$ error [1]',  '.'],
        'struct': ['Structure',  '.'],
        'v_cav_mv': ['Acc. field [MV]', 'o'],
        'phi_s_deg': ['Synch. phase [deg]', 'o'],
        'field_map_factor': [r'$k_e$ [1]', 'o'],
        }

    syn = linac.synch
    # Prep
    if filepath_ref is None:
        filepath_ref = linac.files['project_folder'] \
            + '/results/energy_ref.txt'

    dict_x_data = {
        's': syn.z['abs_array'],
        'elt': range(linac.n_elements),
        }

    # Used when there are too many points (x or y data)
    elts_indexes = linac.get_from_elements('idx', 'out')

    # LW y data
    dict_y_data_lw = {
        'energy': syn.energy['kin_array_mev'],
        'abs_phase': np.rad2deg(syn.phi['abs_array']),
        'beta_synch': syn.energy['beta_array'],
        'v_cav_mv': linac.get_from_elements('acc_field', 'v_cav_mv'),
        'phi_s_deg': linac.get_from_elements('acc_field', 'phi_s_deg'),
        'field_map_factor': linac.get_from_elements('acc_field', 'norm')
        }

    # Coefficient for every error
    dict_err_factor = {
        'energy': 1e3,
        'abs_phase': 1.,
        'beta_synch': 1.,
        }

    # Function to return error between LW and TW data
    def _err(y_d):
        assert y_d in tw.dict_tw_data_table
        y_data_ref = tw.load_tw_results(filepath_ref, y_d)
        y_data = dict_y_data_lw[y_d][elts_indexes]
        err_data = dict_err_factor[y_d] * np.abs(y_data_ref - y_data)
        return err_data

    # Add it to the dict of y data
    dict_errors = {
        'energy_err': _err('energy'),
        'abs_phase_err': _err('abs_phase'),
        'beta_synch_err': _err('beta_synch'),
        }
    dict_y_data_lw.update(dict_errors)

    # Function to plot y_d as a function of x_dat in ax
    def _single_plot(ax, x_dat, y_d, label):
        if y_d == 'struct':
            helper.plot_structure(linac, ax, x_axis=x_dat)

        else:
            # Plot TW data if it was not already done and if it is not an error
            # plot
            if (y_d not in dict_errors) and (
                    y_d in tw.dict_tw_data_table) and (
                    'TW' not in ax.get_legend_handles_labels()[1]):
                x_data_ref = dict_x_data[x_dat]
                y_data_ref = tw.load_tw_results(filepath_ref, y_d)
                x_data_ref, y_data_ref = _reformat(x_data_ref, y_data_ref,
                                                   elts_indexes)
                ax.plot(x_data_ref, y_data_ref, dict_plot[y_d][1], label='TW',
                        c='k', ls='--', linewidth=2.)
            ax.grid(True)
            x_data = dict_x_data[x_dat]
            y_data = dict_y_data_lw[y_d]
            x_data, y_data = _reformat(x_data, y_data, elts_indexes)
            ax.plot(x_data, y_data, dict_plot[y_d][1], label=label, ls='-')

    # Plot
    n_plot = len(y_dat)
    first_axnum = n_plot * 100 + 11
    last_axnum = first_axnum + n_plot
    plt.ion()
    fig, axlist = helper.create_fig_if_not_exist(fignum, range(first_axnum,
                                                               last_axnum))

    for i in range(n_plot):
        y_d = y_dat[i]
        _single_plot(axlist[i], x_dat, y_d, 'LW ' + linac.name)
        axlist[i].set_ylabel(dict_plot[y_d][0])
    axlist[-1].set_xlabel(dict_plot[x_dat][0])
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


def output_cavities(linac, out=True):
    """Output relatable parameters of cavities in list_of_cav."""
    df_cav = pd.DataFrame(columns=(
        'Idx', 'Fail?', 'Comp?', 'Norm', 'phi0', 'Vs', 'phis'))
    list_of_cav = linac.sub_list('FIELD_MAP')
    for i in range(len(list_of_cav)):
        cav = list_of_cav[i]
        df_cav.loc[i] = [cav.idx['in'], cav.status['failed'],
                         cav.status['compensate'], cav.acc_field.norm,
                         np.rad2deg(cav.acc_field.phi_0),
                         cav.acc_field.v_cav_mv, cav.acc_field.phi_s_deg]
    if(out):
        print('\n============================================================')
        helper.printc(linac.name, color='cyan')
        print('\n', df_cav, '\n')
        print('============================================================\n')
    return df_cav


def output_fit(fault_scenario, initial_guess, bounds, out=True):
    """Output relatable parameters of fit."""
    # We change the shape of the bounds if necessary
    if type(bounds) is tuple:
        bounds_fmt = bounds
    else:
        bounds_fmt = (bounds[:, 0], bounds[:, 1])

    list_of_comp = fault_scenario.comp_list['only_cav']
    n_comp = len(list_of_comp)

    list_of_param = ['phi_0', 'Norm']

    dict_param = {
        'phi_0': pd.DataFrame(columns=('Idx', 'Min.', 'Fixed', 'Max.', 'Orig.',
                                       '(var %)')),
        'Norm': pd.DataFrame(columns=('Idx', 'Min.', 'Fixed', 'Max.', 'Orig.',
                                      '(var %)')),
        }
    dict_attribute = {
        'phi_0': lambda acc_f: np.rad2deg(acc_f.phi_0),
        'Norm': lambda acc_f: acc_f.norm,
        }
    dict_guess_bnds = {
        'phi_0': lambda i: [np.rad2deg(initial_guess[i]),
                            np.rad2deg(bounds_fmt[0][i]),
                            np.rad2deg(bounds_fmt[1][i])],
        'Norm': lambda i: [initial_guess[i+n_comp], bounds_fmt[0][i+n_comp],
                           bounds_fmt[1][i+n_comp]]
        }

    for param in list_of_param:
        for i in range(n_comp):
            cav = list_of_comp[i]

            x0_and_bnds = dict_guess_bnds[param](i)
            old = x0_and_bnds[0]
            new = dict_attribute[param](cav.acc_field)
            var = 100. * (new - old) / old

            dict_param[param].loc[i] = [cav.idx['in'], x0_and_bnds[1],
                                        new, x0_and_bnds[2], old, var]

        if(out):
            print('\n========================================================')
            helper.printc(param, color='cyan')
            print('\n', dict_param[param], '\n')
            print('\n========================================================')
    return dict_param
