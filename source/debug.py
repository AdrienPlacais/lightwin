#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021

@author: placais
"""
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
        header = 'Errors on transfer matrix'
        message = 'Error matrix at end of line*1e3:\n' \
            + str(err[-1, 0:2]*1e3) + '\n' + str(err[-1, 2:4]*1e3) \
            + '\nCumulated error:\n' \
            + str(np.linalg.norm(err, axis=0)[0:2]) + '\n' \
            + str(np.linalg.norm(err, axis=0)[2:4]) \
            + '\n\nCumulated error:\n' \
            + str(np.linalg.norm(err, axis=0)[0:2]) + '\n' \
            + str(np.linalg.norm(err, axis=0)[2:4]) \
            + '\n\nTot error:\n' + str(np.linalg.norm(err))
        helper.printd(message, header=header)
    return err, z_err


def plot_transfer_matrices(accelerator, transfer_matrix):
    """
    Plot the transfer matrix components of TraceWin and LightWin.

    Parameters
    ----------
    accelerator: Accelerator object
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

    r_zz_tot_ref = tw.load_transfer_matrices(filepath_ref)

    err, z_err = compute_error_transfer_matrix(r_zz_tot, r_zz_tot_ref,
                                               flag_output=False)

    axnumlist = range(221, 225)
    _, axlist = helper.create_fig_if_not_exist(26, axnumlist)
    labels = {
        'TW': ['TW', '', '', ''],
        'LW': [accelerator.name, '', '', ''],
        'x': ['', '', 's [m]', 's [m]'],
        'y': [r'$R_{11}$', r'$R_{12}$', r'$R_{21}$', r'$R_{22}$'],
        }
    lims = {
        0: np.array([-1.3, 1.4]),
        1: np.array([-1.9, 1.9]),
        2: np.array([-1., 1.2]),
        3: np.array([-1.4, 1.4]),
        }

    if 'TW' not in axlist[0].get_legend_handles_labels()[1]:
        for i in range(4):
            axlist[i].plot(r_zz_tot_ref[:, 0], r_zz_tot_ref[:, i+1],
                           label=labels['TW'][i], ls='--', c='k')

    for i in range(4):
        axlist[i].plot(r_zz_tot[:, 0], r_zz_tot[:, i+1],
                       label=labels['LW'][i])
        axlist[i].set_xlabel(labels['x'][i])
        axlist[i].set_ylabel(labels['y'][i])
        axlist[i].grid(True)
        axlist[i].set_ylim(lims[i])

    axlist[0].legend()

    axlist = []
    _, axlist = helper.create_fig_if_not_exist(260, axnumlist)

    for i in range(4):
        axlist[i].plot(z_err, err[:, i], label=labels['LW'][i])
        axlist[i].set_xlabel(labels['x'][i])
        axlist[i].set_ylabel(r'$\varepsilon$' + labels['y'][i])
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


def _create_plot_dicts():
    # [label, marker]
    dict_plot = {
            's': ['Synch. position [m]',
                  {'marker': None}],
            'elt': ['Element number',
                    {'marker': None}],
            'energy': ['Beam energy [MeV]',
                       {'marker': None}],
            'energy_err': ['Log of abs. error [1]',
                           {'marker': None}],
            'abs_phase': ['Beam phase [deg]',
                          {'marker': None}],
            'abs_phase_err': ['Log of phase error [1]',
                              {'marker': None}],
            'beta_synch': [r'Synch. $\beta$ [1]',
                           {'marker': None}],
            'beta_synch_err': [r'Abs. $\beta$ error [1]',
                               {'marker': None}],
            'struct': ['Structure',
                       {'marker': None}],
            'v_cav_mv': ['Acc. field [MV]',
                         {'marker': 'o'}],
            'phi_s_deg': ['Synch. phase [deg]',
                          {'marker': 'o'}],
            'field_map_factor': [r'$k_e$ [1]',
                                 {'marker': 'o'}],
        }

    dict_x_data = {
        's': lambda lin: lin.synch.z['abs_array'],
        'elt': lambda lin: np.array(range(lin.elements['n'])),
        }

    # LW y data
    dict_y_data_lw = {
        'energy': lambda lin: lin.synch.energy['kin_array_mev'],
        'abs_phase': lambda lin: np.rad2deg(lin.synch.phi['abs_array']),
        'beta_synch': lambda lin: lin.synch.energy['beta_array'],
        'v_cav_mv': lambda lin:
            lin.get_from_elements('acc_field', 'cav_params', 'v_cav_mv'),
        'phi_s_deg': lambda lin:
            lin.get_from_elements('acc_field', 'cav_params', 'phi_s_deg'),
        'field_map_factor': lambda lin:
            lin.get_from_elements('acc_field', 'norm')
        }

    dict_err_factor = {
        'energy': 1,
        'abs_phase': 1.,
        'beta_synch': 1.,
        }

    all_dicts = {
        'plot': dict_plot,
        'x_data': dict_x_data,
        'y_data_lw': dict_y_data_lw,
        'err_factor': dict_err_factor,
        # 'errors': dict_errors,
        }

    return all_dicts


def compare_with_tracewin(linac, x_dat='s', y_dat=None, filepath_ref=None,
                          fignum=21):
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
    if y_dat is None:
        y_dat = ['energy', 'energy_err', 'struct']
    if filepath_ref is None:
        filepath_ref = linac.files['project_folder'] \
            + '/results/energy_ref.txt'

    dicts = _create_plot_dicts()

    elts_indexes = linac.get_from_elements('idx', 's_out')

    def _err(y_d, diff):
        assert y_d in tw.dict_tw_data_table
        y_data_ref = tw.load_tw_results(filepath_ref, y_d)
        y_data = dicts['y_data_lw'][y_d](linac)[elts_indexes]
        if diff == 'abs':
            err_data = dicts['err_factor'][y_d] * np.abs(y_data_ref - y_data)
        elif diff == 'rel':
            err_data = dicts['err_factor'][y_d] * (y_data_ref - y_data)
        elif diff == 'log':
            err_data = dicts['err_factor'][y_d] * np.log10(y_data / y_data_ref)
        return err_data
    # Add it to the dict of y data
    dict_errors = {
        'energy_err': lambda lin: _err('energy', diff='log'),
        'abs_phase_err': lambda lin: _err('abs_phase', diff='log'),
        'beta_synch_err': lambda lin: _err('beta_synch', diff='abs'),
        }
    dicts['errors'] = dict_errors
    dicts['y_data_lw'].update(dict_errors)

    # Plot
    first_axnum = len(y_dat) * 100 + 11
    _, axlist = helper.create_fig_if_not_exist(
        fignum, range(first_axnum, first_axnum + len(y_dat)), sharex=True,
        )

    for i, y_d in enumerate(y_dat):
        _single_plot(axlist[i], [x_dat, y_d], dicts, filepath_ref, linac)
        axlist[i].set_ylabel(dicts['plot'][y_d][0])
    axlist[-1].set_xlabel(dicts['plot'][x_dat][0])
    axlist[0].legend()


def _single_plot(axx, xydata, dicts, filepath_ref, linac, plot_section=True):
    """Plot proper data in proper subplot."""
    x_dat = xydata[0]
    y_d = xydata[1]
    elts_indexes = linac.get_from_elements('idx', 's_out')
    if plot_section:
        helper.plot_section(linac, axx, x_axis=x_dat)
    if y_d == 'struct':
        helper.plot_structure(linac, axx, x_axis=x_dat)

    else:
        # Plot TW data if it was not already done and if it is not an error
        # plot
        if (y_d not in dicts['errors']) and (y_d in tw.dict_tw_data_table
                                             ) and (
                'TW' not in axx.get_legend_handles_labels()[1]):
            x_data_ref = dicts['x_data'][x_dat](linac)
            y_data_ref = tw.load_tw_results(filepath_ref, y_d)
            x_data_ref, y_data_ref = _reformat(x_data_ref, y_data_ref,
                                               elts_indexes)
            axx.plot(x_data_ref, y_data_ref, label='TW',
                     c='k', ls='--', linewidth=2., **dicts['plot'][y_d][1])
        axx.grid(True)
        x_data = dicts['x_data'][x_dat](linac)
        y_data = dicts['y_data_lw'][y_d](linac)
        x_data, y_data = _reformat(x_data, y_data, elts_indexes)
        # dicts['plot'][y_d][1] is a dict that looks like to:
        # {'marker': '+', 'linewidth': 5}
        # ** (**kwargs) unpacks it to:
        # marker='+', linewidth=5
        label = 'LW ' + linac.name
        if not (linac.name in ['Working', 'Broken'] and
                label in axx.get_legend_handles_labels()[1]):
            axx.plot(x_data, y_data, label=label, ls='-',
                     **dicts['plot'][y_d][1])


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
    _, axx = helper.create_fig_if_not_exist(41, [111])
    axx = axx[0]
    axx.set_xlabel(r'$\delta z$ [mm]')

    # Set proper y axis and access to proper y data
    if x_axis == 'z':
        axx.set_xlabel(r'$\delta z$ [mm]')
        x_data = {
            'tw': lambda element, i: element['z(mm)'][i],
            'lw': lambda part: part.phase_space['z_array'] * 1e3,
                }

    elif x_axis == 'phase':
        axx.set_xlabel(r'$\phi$ [deg]')
        x_data = {
            'tw': lambda element, i: element['Phase(deg)'][i],
            'lw': lambda part: np.rad2deg(part.phase_space['phi_array_rad']),
                }

    else:
        raise IOError('Wrong x_axis argument in compare_phase_space.')

    # Set proper y axis and access to proper y data
    if y_axis == 'E':
        axx.set_ylabel(r'$E$ [MeV]')
        y_data = {
            'tw': lambda element, i: element['Energy(MeV)'][i],
            'lw': lambda part: part.energy['kin_array_mev'],
                }

    elif y_axis == 'dp/p':
        axx.set_ylabel(r'$dp/p$ [%]')
        y_data = {
            'tw': lambda element, i: helper.mrad_and_mev_to_delta(
                element["z'(mrad)"][i], element['Energy(MeV)'][i], E_rest_MeV),
            'lw': lambda part: part.phase_space['delta_array'] * 100.,
                }

    elif y_axis == "z'":
        axx.set_ylabel(r"$z'$ [mrad]")
        y_data = {
            'tw': lambda element, i: element["z'(mrad)"][i],
            'lw': lambda part: part.phase_space['delta_array']
            * part.energy['gamma_array']**-2 * 1e3,
                }

    else:
        raise IOError('Wrong y_axis argument in compare_phase_space.')

    axx.grid(True)

    # Load TW data
    partran_data = load_phase_space(accelerator)
    n_part = partran_data[0]['x(mm)'].size

    # Plot TW data
    for element in partran_data:
        for i in range(n_part):
            if i in idx_of_part_to_plot:
                axx.scatter(x_data['tw'](element, i), y_data['tw'](element, i),
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
    idx = accelerator.get_from_elements('idx', 's_in')

    i = 0
    for part in particle_list:
        if i in idx_of_part_to_plot:
            helper.plot_pty_with_data_tags(axx, x_data['lw'](part),
                                           y_data['lw'](part), idx, tags=True)
        i += 1

    # TODO: remove when useless
    accelerator.particle_list = particle_list
    accelerator.partran_data = partran_data


def output_cavities(linac, out=False):
    """Output relatable parameters of cavities in list_of_cav."""
    df_cav = pd.DataFrame(columns=(
        'Name', 'Status?', 'Norm', 'phi0 abs', 'phi_0 rel', 'Vs',
        'phis'))
    full_list_of_cav = linac.elements_of('FIELD_MAP')

    for i, cav in enumerate(full_list_of_cav):
        df_cav.loc[i] = [cav.info['name'], cav.info['status'],
                         cav.acc_field.norm,
                         np.rad2deg(cav.acc_field.phi_0['abs']),
                         np.rad2deg(cav.acc_field.phi_0['rel']),
                         cav.acc_field.cav_params['v_cav_mv'],
                         cav.acc_field.cav_params['phi_s_deg']]
    df_cav.round(decimals=3)

    # Output only the cavities that have changed
    if 'Fixed' in linac.name:
        df_out = pd.DataFrame(columns=(
            'Name', 'Status?', 'Norm', 'phi0 abs', 'phi_0 rel', 'Vs',
            'phis'))
        i = 0
        for c in full_list_of_cav:
            if c.info['status'] != 'nominal':
                i += 1
                df_out.loc[i] = df_cav.loc[full_list_of_cav.index(c)]
        if out:
            helper.printd(df_out, header=linac.name)
    return df_cav


def _create_output_fit_dicts():
    dict_param = {
        'phi_0_rel': pd.DataFrame(columns=('Name', 'Status', 'Min.', 'Max.',
                                           'Fixed', 'Orig.', '(var %)')),
        'phi_0_abs': pd.DataFrame(columns=('Name', 'Status', 'Min.', 'Max.',
                                           'Fixed', 'Orig.', '(var %)')),
        'Norm': pd.DataFrame(columns=('Name', 'Status', 'Min.', 'Max.',
                                      'Fixed', 'Orig.', '(var %)')),
        }
    dict_attribute = {
        'phi_0_rel': lambda cav: np.rad2deg(cav.acc_field.phi_0['rel']),
        'phi_0_abs': lambda cav: np.rad2deg(cav.acc_field.phi_0['abs']),
        'Norm': lambda cav: cav.acc_field.norm,
        }
    # Hypothesis: the first guesses for the phases are the phases of the
    # reference cavities
    dict_guess_bnds = {
        'phi_0_rel':
            lambda f, i:
                [np.rad2deg(f.info['bounds'][0][i]),
                 np.rad2deg(f.info['bounds'][1][i])
                 ],
        'phi_0_abs':
            lambda f, i:
                [np.rad2deg(f.info['bounds'][0][i]),
                 np.rad2deg(f.info['bounds'][1][i])],
        'Norm':
            lambda f, i:
                [f.info['bounds'][0][i+len(f.comp['l_cav'])],
                 f.info['bounds'][1][i+len(f.comp['l_cav'])]]
        }

    all_dicts = {
        'param': dict_param,
        'attribute': dict_attribute,
        'guess_bnds': dict_guess_bnds,
        }

    return all_dicts


def output_fit(fault_scenario, out_detail=False, out_compact=True):
    """Output relatable parameters of fit."""
    dicts = _create_output_fit_dicts()

    shift_i = 0
    for f in fault_scenario.faults['l_obj']:
        # We change the shape of the bounds if necessary
        if not isinstance(f.info['bounds'], tuple):
            f.info['bounds'] = (f.info['bounds'][:, 0], f.info['bounds'][:, 1])

        # Get list of compensating cavities, and their original counterpart in
        # the reference linac
        ref_equiv = [
            f.ref_lin.elements['list'][cav.idx['element']]
            for cav in f.comp['l_cav']
            ]

        for param in dicts['param']:
            dicts['param'][param].loc[shift_i] = \
                ['----', '----------', None, None, None, None, None]
            for i, cav in enumerate(f.comp['l_cav']):
                bnds = dicts['guess_bnds'][param](f, i)
                old = dicts['attribute'][param](ref_equiv[i])
                new = dicts['attribute'][param](cav)
                var = 100. * (new - old) / old

                dicts['param'][param].loc[i + shift_i + 1] =\
                    [cav.info['name'], cav.info['status'], bnds[0], bnds[1],
                     new, old, var]
        shift_i += i + 2

    if out_detail:
        for param in dicts['param']:
            helper.printd(dicts['param'][param].round(3), header=param)

    compact = pd.DataFrame(columns=('Name', 'Status', 'Norm', '(var %)',
                                    'phi_0 (rel)', 'phi_0 (abs)'))
    for i in range(dicts['param']['Norm'].shape[0]):
        compact.loc[i] = [
            dicts['param']['Norm']['Name'][i],
            dicts['param']['Norm']['Status'][i],
            dicts['param']['Norm']['Fixed'][i],
            dicts['param']['Norm']['(var %)'][i],
            dicts['param']['phi_0_rel']['Fixed'][i],
            dicts['param']['phi_0_abs']['Fixed'][i],
                          ]
    if out_compact:
        helper.printd(compact.round(3), header='Fit compact resume')

    return dicts['param']


def output_fit_progress(count, obj, final=False):
    """Output the evolution of the objective, etc."""
    if count == 0:
        print('=============================================================' +
              '=============================================================' +
              '=============')
        print('iter', '      phi', '    energy', '      M_11', '      M_12',
              '      M_21', '      M_22', '       phi', '    energy',
              '      M_11', '      M_12', '      M_21', '      M_22')
        print('=============================================================' +
              '=============================================================' +
              '=============')
    print(count, end='\t')
    max_width = 10
    precision = 3
    for num in obj:
        # print(str(round(num, 3)), end=' ')
        print(f"{num: {max_width}.{precision}}", end=' ')
    print(' ')
    if final:
        print('=============================================================' +
              '=============================================================' +
              '=============')
