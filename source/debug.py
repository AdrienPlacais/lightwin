#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021.

@author: placais
TODO merge dict entries for phase space, ie alpha['z'] instead of 'alpha_z'
TODO ellipse plot could be better
"""
from os import listdir
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
from palettable.colorbrewer.qualitative import Set1_9
from cycler import cycler
from dicts_output import d_markdown, d_plot_kwargs
import helper
import tracewin_interface as tw

font = {'family': 'serif',
        'size': 20}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Set1_9.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def compute_error_transfer_matrix(transf_mat, transf_mat_ref, output=False):
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
                                y=transf_mat_ref[:, i + 1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = f_interp(z_err) - transf_mat[:, i + 1]
            # err[:, i] = np.log10(transf_mat[:, i + 1] / f_interp(z_err))

    else:
        z_err = transf_mat_ref[:, 0]
        err = np.full((n_z_ref, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=transf_mat[:, 0],
                                y=transf_mat[:, i + 1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = transf_mat_ref[:, i + 1] - f_interp(z_err)
            # err[:, i] = np.log10(transf_mat_ref[:, i + 1] / f_interp(z_err))

    if output:
        header = "Errors on transfer matrix"
        message = f"""
            Error matrix at end of line*1e3:
            {err[-1, 0:2] * 1e3}
            {err[-1, 2:4] * 1e3}

            Cumulated error:
            {np.linalg.norm(err, axis=0)[0:2]}
            {np.linalg.norm(err, axis=0)[2:4]}

            Cumulated error:
            {np.linalg.norm(err, axis=0)[0:2]}
            {np.linalg.norm(err, axis=0)[2:4]}

            Tot error:
            {np.linalg.norm(err)}
            """
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

    z_pos = accelerator.synch.pos['z_abs']
    n_z = z_pos.shape[0]

    transfer_matrix = accelerator.transf_mat['tm_cumul']

    # Change shape of calculated transfer matrix to match the ref one
    # i.e.: 1st column is z, 2nd 3rd 4th and 5th are matrix components
    # r_zz_tot = accelerator.transfer_matrix_cumul.reshape((n_z, 4))
    r_zz_tot = transfer_matrix.reshape((n_z, 4))
    r_zz_tot = np.hstack((np.expand_dims(z_pos, 1), r_zz_tot))

    r_zz_tot_ref = tw.load_transfer_matrices(filepath_ref)

    err, z_err = compute_error_transfer_matrix(r_zz_tot, r_zz_tot_ref,
                                               output=False)

    axnumlist = range(221, 225)
    _, axlist = helper.create_fig_if_not_exist(31, axnumlist)
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
            axlist[i].plot(r_zz_tot_ref[:, 0], r_zz_tot_ref[:, i + 1],
                           label=labels['TW'][i], ls='--', c='k')

    for i in range(4):
        axlist[i].plot(r_zz_tot[:, 0], r_zz_tot[:, i + 1],
                       label=labels['LW'][i])
        axlist[i].set_xlabel(labels['x'][i])
        axlist[i].set_ylabel(labels['y'][i])
        axlist[i].grid(True)
        axlist[i].set_ylim(lims[i])

    axlist[0].legend()

    axlist = []
    _, axlist = helper.create_fig_if_not_exist(310, axnumlist)

    for i in range(4):
        axlist[i].plot(z_err, err[:, i], label=labels['LW'][i])
        axlist[i].set_xlabel(labels['x'][i])
        axlist[i].set_ylabel(r'$\varepsilon$' + labels['y'][i])
        axlist[i].set_yscale('log')
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


# FIXME Some pieces of code are repeated, can do better
def compare_with_tracewin(linac, x_str='z_abs', l_y_str=None,
                           filepath_ref=None, fignum=21):
    """Plot data calculated by TraceWin and LightWin."""
    plot_section = True
    # Default plot
    if l_y_str is None:
        l_y_str = ['w_kin', 'w_kin_err', 'struct']

    if filepath_ref is None:
        filepath_ref = linac.get('project_folder') + '/results/energy_ref.txt'

    # Prep some data common to all plots
    x_dat = linac.get(x_str, to_deg=True)
    elts_indexes = linac.get('s_out')

    # Prep figure and axes
    n_plots = len(l_y_str)
    axnum = 100 * n_plots + 11
    axnum = range(axnum, axnum + n_plots)
    _, axx = helper.create_fig_if_not_exist(fignum, axnum, sharex=True)

    # Get data and kwargs
    for i, y_str in enumerate(l_y_str):
        if plot_section:
            helper.plot_section(linac, axx[i], x_axis=x_str)

        if y_str == 'struct':
            helper.plot_structure(linac, axx[i], x_axis=x_str)
            continue

        l_y_dat = []
        l_kwargs = []

        # Plot error data?
        plot_error = y_str[-4:] == '_err'

        # Plot TW data?
        label_tw = 'TW'
        plot_tw = not plot_error and y_str in tw.d_tw_data_table \
                and label_tw not in axx[i].get_legend_handles_labels()[1]

        # Replot Working and Broken every time?
        replot_lw = False
        label_lw = 'LW ' + linac.name
        plot_lw = not plot_error \
                and (replot_lw
                     or linac.name not in ['Working', 'Broken']
                     or label_lw not in axx[i].get_legend_handles_labels()[1]
                    )

        if plot_error:
            diff = 'abs'
            if 'abs_phase' in y_str or 'w_kin' in y_str:
                diff = 'log'
            l_y_dat.append(_err(linac, y_str[:-4], filepath_ref, diff=diff))
            l_kwargs.append(
                d_plot_kwargs[y_str] | {'label': linac.name + 'err. w/ TW'}
            )

        if plot_tw:
            l_y_dat.append(tw.load_tw_results(filepath_ref, y_str))
            l_kwargs.append(
                d_plot_kwargs[y_str] | {'label': label_tw, 'c': 'k', 'lw': 2.,
                                        'ls': '--'}
            )

        if plot_lw:
            # LightWin
            l_y_dat.append(linac.get(y_str, to_deg=True))
            l_kwargs.append(
                d_plot_kwargs[y_str] | {'label': label_lw}
            )

        for y_dat, kwargs in zip(l_y_dat, l_kwargs):
            # Downsample x or y if necessary
            x_plot, y_plot = _reformat(x_dat, y_dat, elts_indexes)

            axx[i].plot(x_plot, y_plot, **kwargs)

        axx[i].set_ylabel(d_markdown[y_str])
        axx[i].grid(True)

    axx[0].legend()
    axx[-1].set_xlabel(d_markdown[x_str])

def _err(linac, y_str, filepath_ref, diff='abs'):
    """Calculate error between linac (LW) and TW."""
    assert y_str in tw.d_tw_data_table
    elts_indexes = linac.get('s_out')

    # Set up a scale (for example if the error is very small)
    d_err_scales = {
        'example': 1e3,
    }
    scale = d_err_scales.get(y_str, 1.)
    # this is the default value ----'

    d_diff = {'abs': lambda ref, new: scale * np.abs(ref - new),
              'rel': lambda ref, new: scale * (ref - new),
              'log': lambda ref, new: scale * np.log10(np.abs(new / ref)),
             }

    y_data_tw = tw.load_tw_results(filepath_ref, y_str)
    y_data_lw = linac.get(y_str)[elts_indexes]
    return d_diff[diff](y_data_tw, y_data_lw)

    # Old piece of code for autoscale
        # FIXME does not work on plots without legend...
        # if ignore_broken_ylims:
            # if 'Fixed' in linac.name:
                # lines_labels = axx.get_legend_handles_labels()
                # try:
                    # idx_to_ignore = lines_labels[1].index('LW Broken')
                    # lines_labels[0].pop(idx_to_ignore)
                # except ValueError:
                    # pass
                # _autoscale_based_on(axx, lines_labels[0])


def _autoscale_based_on(axx, lines):
    axx.dataLim = mtransforms.Bbox.unit()
    for line in lines:
        datxy = np.vstack(line.get_data()).T
        axx.dataLim.update_from_data_xy(datxy, ignore=False)
    axx.autoscale_view()


# TODO: move dicts into the function dedicated to dicts creation
def plot_ellipse_emittance(axx, accelerator, idx, phase_space="w"):
    """Plot the emittance ellipse and highlight interesting data."""
    # Extract Twiss and emittance at the index idx
    twi = accelerator.get("twiss_" + phase_space)[idx]
    eps = accelerator.get("eps_ " + phase_space)[idx]

    # Compute ellipse dimensions; ellipse equation:
    # Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0
    d_eq = {"A": twi[2], "B": 2. * twi[0], "C": twi[1], "D": 0., "E": 0.,
            "F": -eps}

    # Plot ellipse
    d_colors = {"Working": "k",
                "Broken": "r",
                "Fixed": "g"}
    color = d_colors[accelerator.name.split(" ")[0]]
    plot_kwargs = {"c": color}
    helper.plot_ellipse(axx, d_eq, **plot_kwargs)

    # Set proper labels
    d_xlabel = {"z": r"Position $z$ [mm]",
                "zdelta": r"Position $z$ [mm]",
                "w": r"Phase $\phi$ [deg]"}
    axx.set_xlabel(d_xlabel[phase_space])

    d_ylabel = {"z": r"Speed $z'$ [%]",
                "zdelta": r"Speed $\delta p/p$ [mrad]",
                "w": r"Energy $W$ [MeV]"}
    axx.set_ylabel(d_ylabel[phase_space])

    form = "{:.3g}"
    # Max phase
    maxi_phi = np.sqrt(eps * twi[1])
    line = axx.axvline(maxi_phi, c='b')
    axx.axhline(-twi[0] * np.sqrt(eps / twi[1]), c=line.get_color())
    axx.get_xticklabels().append(
        plt.text(1.005 * maxi_phi, .05, form.format(maxi_phi),
                 va="bottom", rotation=90.,
                 transform=axx.get_xaxis_transform(), c=line.get_color())
    )

    # Max energy
    maxi_w = np.sqrt(eps * twi[2])
    line = axx.axhline(maxi_w)
    axx.axvline(-twi[0] * np.sqrt(eps / twi[2]), c=line.get_color())
    axx.get_yticklabels().append(
        plt.text(.005, .95 * maxi_w, form.format(maxi_w), va="top",
                 rotation=0.,
                 transform=axx.get_yaxis_transform(), c=line.get_color())
    )

    axx.grid(True)


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


def output_cavities(linac, out=False):
    """Output relatable parameters of cavities in list_of_cav."""
    columns=('elt_name', 'status', 'k_e', 'phi_0_abs', 'phi_0_rel', 'v_cav_mv',
             'phi_s')
    df_cav = pd.DataFrame(columns=columns)

    full_list_of_cav = linac.elements_of(nature='FIELD_MAP')

    for i, cav in enumerate(full_list_of_cav):
        df_cav.loc[i] = cav.get(*columns, to_deg=True)
    df_cav.round(decimals=3)

    # Output only the cavities that have changed
    if 'Fixed' in linac.name:
        df_out = pd.DataFrame(columns=columns)
        i = 0
        for cav in full_list_of_cav:
            if 'compensate' in cav.get('status'):
                i += 1
                df_out.loc[i] = df_cav.loc[full_list_of_cav.index(cav)]
        if out:
            helper.printd(df_out, header=linac.name)
    return df_cav


def _create_output_fit_dicts():
    col = ('Name', 'Status', 'Min.', 'Max.', 'Fixed', 'Orig.', '(var %)')
    d_pd = {'phi_0_rel': pd.DataFrame(columns=col),
            'phi_0_abs': pd.DataFrame(columns=col),
            'k_e': pd.DataFrame(columns=col),
    }
    # Hypothesis: the first guesses for the phases are the phases of the
    # reference cavities
    d_x_lim = {
        'phi_0_rel': lambda f, i: [np.rad2deg(f.info['X_lim'][0][i]),
                                   np.rad2deg(f.info['X_lim'][1][i])],
        'phi_0_abs': lambda f, i: [np.rad2deg(f.info['X_lim'][0][i]),
                                   np.rad2deg(f.info['X_lim'][1][i])],
        'k_e': lambda f, i: [f.info['X_lim'][0][i + len(f.comp['l_cav'])],
                             f.info['X_lim'][1][i + len(f.comp['l_cav'])]]
    }

    all_dicts = {'d_pd': d_pd,
                 'd_X_lim': d_x_lim}

    return all_dicts


def output_fit(fault_scenario, out_detail=False, out_compact=True):
    """Output relatable parameters of fit."""
    dicts = _create_output_fit_dicts()
    d_pd = dicts['d_pd']

    shift_i = 0
    i = None
    for __f in fault_scenario.faults['l_obj']:
        # We change the shape of the bounds if necessary
        if not isinstance(__f.info['X_lim'], tuple):
            __f.info['X_lim'] = (__f.info['X_lim'][:, 0],
                                 __f.info['X_lim'][:, 1])

        # Get list of compensating cavities, and their original counterpart in
        # the reference linac
        idx_equiv = [cav.idx['elt_idx'] for cav in __f.comp['l_cav']]
        ref_equiv = [__f.ref_lin.elts[idx] for idx in idx_equiv]

        for key, val in d_pd.items():
            val.loc[shift_i] = \
                ['----', '----------', None, None, None, None, None]
            for i, cav in enumerate(__f.comp['l_cav']):
                x_lim = dicts['d_X_lim'][key](__f, i)
                old = ref_equiv[i].get(key, to_deg=True)
                new = cav.get(key, to_deg=True)
                if old is None or new is None:
                    var = None
                else:
                    var = 100. * (new - old) / old

                val.loc[i + shift_i + 1] = \
                    [cav.get('elt_name'), cav.get('status'), x_lim[0], x_lim[1],
                     new, old, var]
        shift_i += i + 2

    if out_detail:
        for key, val in d_pd.items():
            helper.printd(val.round(3), header=key)

    compact = pd.DataFrame(columns=('Name', 'Status', 'k_e', '(var %)',
                                    'phi_0 (rel)', 'phi_0 (abs)'))
    for i in range(d_pd['k_e'].shape[0]):
        compact.loc[i] = [
            d_pd['k_e']['Name'][i],
            d_pd['k_e']['Status'][i],
            d_pd['k_e']['Fixed'][i],
            d_pd['k_e']['(var %)'][i],
            d_pd['phi_0_rel']['Fixed'][i],
            d_pd['phi_0_abs']['Fixed'][i],
        ]
    if out_compact:
        helper.printd(compact.round(3), header='Fit compact resume')

    return d_pd


def output_fit_progress(count, obj, l_label, final=False):
    """Output the evolution of the objective, etc."""
    single_width = 10
    precision = 3
    total_width = (len(obj) + 1) * (single_width + precision)

    if count == 0:
        n_param = len(l_label)
        n_cav = len(obj) // n_param
        print(''.center(total_width, '='))
        print(" iteration", end=' ')
        for i in range(n_cav):
            for header in l_label:
                print(f"{header: >{single_width}}", end=' ')
        print('\n' + ''.center(total_width, '='))

    print(f"{count: {single_width}}", end=' ')
    for num in obj:
        print(f"{num: {single_width}.{precision}e}", end=' ')
    print(' ')
    if final:
        print(''.center(total_width, '='))


def plot_fit_progress(hist_f, l_label):
    """Plot the evolution of the objective functions w/ each iteration."""
    _, axx = helper.create_fig_if_not_exist(32, [111])
    axx = axx[0]

    # Number of objectives, number of evaluations
    n_f = len(l_label)
    n_iter = len(hist_f)
    iteration = np.linspace(0, n_iter - 1, n_iter)

    __f = np.empty([n_f, n_iter])
    for i in range(n_iter):
        __f[:, i] = np.abs(hist_f[i] / hist_f[0])

    for j, label in enumerate(l_label):
        axx.plot(iteration, __f[j], label=label)

    axx.grid(True)
    axx.legend()
    axx.set_xlabel("Iteration #")
    axx.set_ylabel("Relative variation of error")
    axx.set_yscale('log')
