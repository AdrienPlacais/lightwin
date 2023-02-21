#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:50:44 2021.

@author: placais
TODO merge dict entries for phase space, ie alpha['z'] instead of 'alpha_z'
TODO ellipse plot could be better
"""
import os

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd

import util.tracewin_interface as tw
from util.dicts_output import d_markdown, d_plot_kwargs
from util import helper



# TODO modernize
def compute_error_transfer_matrix(t_m, t_m_ref, output=False):
    """Compute and output error between transfer matrix and ref."""
    n_z = t_m.shape[0]
    n_z_ref = t_m_ref.shape[0]

    # We calculate error by interpolating the tab with most points on the one
    # with least points.
    kind = 'linear'
    bounds_error = False
    fill_value = 'extrapolate'

    if n_z < n_z_ref:
        z_err = t_m[:, 0]
        err = np.full((n_z, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=t_m_ref[:, 0],
                                y=t_m_ref[:, i + 1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = f_interp(z_err) - t_m[:, i + 1]

    else:
        z_err = t_m_ref[:, 0]
        err = np.full((n_z_ref, 4), np.NaN)
        for i in range(4):
            f_interp = interp1d(x=t_m[:, 0],
                                y=t_m[:, i + 1],
                                kind=kind, bounds_error=bounds_error,
                                fill_value=fill_value)
            err[:, i] = t_m_ref[:, i + 1] - f_interp(z_err)

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



def load_phase_space(accelerator):
    """
    Load Partran phase-space data.

    Phase-space files are obtained with:
        Input data & Beam: Partran
        Phase spaces or beam distributions: Output at element n
        Then save all particle as ASCII.
    """
    folder = os.path.join(accelerator.get('project_folder'),
                          'results/phase_space/')
    file_type = ['txt']
    file_list = []

    for file in os.listdir(folder):
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
    columns = ('elt_name', 'status', 'k_e', 'phi_0_abs', 'phi_0_rel',
               'v_cav_mv', 'phi_s')
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
            val.loc[shift_i] = ['----', '----------',
                                None, None, None, None, None]
            for i, cav in enumerate(__f.comp['l_cav']):
                x_lim = dicts['d_X_lim'][key](__f, i)
                old = ref_equiv[i].get(key, to_deg=True)
                new = cav.get(key, to_deg=True)
                if old is None or new is None:
                    var = None
                else:
                    var = 100. * (new - old) / old

                val.loc[i + shift_i + 1] = \
                    [cav.get('elt_name'), cav.get('status'), x_lim[0],
                     x_lim[1], new, old, var]
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
    """Output the evolution of the objectives."""
    single_width = 30
    precision = 3
    str_iter = ' iter.'
    width_separation = 3
    end = " " * width_separation
    lengths = [len(label) + width_separation for label in l_label]
    total_width = len(str_iter) + sum(lengths)

    n_param = len(l_label)
    n_cav = len(obj) // n_param

    if count == 0:
        print('=' * total_width)
        print(str_iter, end=end)
        for i in range(n_cav):
            for header in l_label:
                print(f"{header: >{len(header)}}", end=end)
        print('\n' + '=' * total_width)

    print(f"{count: {len(str_iter)}}", end=end)
    for num, length in zip(obj, lengths):
        out = f"{num: {length - precision - width_separation + 3}.{precision}e}"
        print(out, end=end)
    print(' ')
    if final:
        print(''.center(total_width, '='))
