#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:52:26 2023.

@author: placais

Routines to evaluate the quality of the new settings for the linac.

# TODO a lot of things to update with SimulationOutput!!
"""
import numpy as np
import matplotlib.pyplot as plt

from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler

from core.beam_parameters import mismatch_factor
from core.accelerator import Accelerator
import visualization.plot

font = {'family': 'serif',
        'size': 20}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=cycler('color', Dark2_8.mpl_colors))
plt.rc('mathtext', fontset='cm')


def fred_tests(lin_ref: Accelerator, lin_fix: Accelerator,
               multipart: bool = True, plot: bool = True) -> dict[[str], bool]:
    """
    Check if the new settings are ok.

    The tests are:
        - lost power shall be null;
        - the RMS emittances shall not grow of more than 20% along the linac;
        - the maximum of 99% emittances (fixed) shall not exceed the nominal
          maximum of 99% emittances by more than 30%.

    This routine simply returns a dict containing a boolean telling if these
    tests´ were successfully passed or not.
    """
    if multipart:
        ref_results = lin_ref.tracewin_simulation.results_multipart
        fix_results = lin_fix.tracewin_simulation.results_multipart
    else:
        ref_results = lin_ref.tracewin_simulation.results_envelope
        fix_results = lin_fix.tracewin_simulation.results_envelope

    acceptable_limits = []

    test_flags = {'Powlost': True, 'ex': True, 'ey': True, 'ep': True,
                  'ex99': True, 'ey99': True, 'ep99': True}

    # Power loss test
    pow_lost = fix_results['Powlost']
    if pow_lost[-1] > 1e-10:
        test_flags['Powlost'] = False
    acceptable_limits.append({'Powlost': {'max': None, 'min': None}})

    # RMS emittances test
    eps_rms = np.column_stack(
        (fix_results['ex'], fix_results['ey'], fix_results['ep']))
    var_rms = 100. * (eps_rms - eps_rms[0, :]) / eps_rms[0, :]

    rms_emittance_limits = {}
    for i, key in enumerate(['ex', 'ey', 'ep']):
        rms_emittance_limits[key] = {'max': 1.2 * ref_results[key],
                                     'min': None}
        if np.any(var_rms[:, i] > 20.):
            test_flags[key] = False
    acceptable_limits.append(rms_emittance_limits)

    # 99% emittances test
    eps99_ref = np.max(np.column_stack(
        (ref_results['ex99'], ref_results['ey99'], ref_results['ep99'])
    ), axis=0)
    eps99_fix = np.max(np.column_stack(
        (fix_results['ex99'], fix_results['ey99'], fix_results['ep99'])
    ), axis=0)

    eps99_limits = {}
    for i, key in enumerate(['ex99', 'ey99', 'ep99']):
        eps99_limits[key] = {'max': 1.3 * np.max(ref_results[key]),
                             'min': None}
        if eps99_fix[i] > 1.3 * eps99_ref[i]:
            test_flags[key] = False
    acceptable_limits.append(eps99_limits)

    if plot:
        tests_to_plot_together = [['Powlost'],
                                  ['ex', 'ey', 'ep'],
                                  ['ex99', 'ey99', 'ep99']]
        reference_values = [{key: ref_results[key] for key in single_plot}
                            for single_plot in tests_to_plot_together]
        fixed_values = [{key: fix_results[key] for key in single_plot}
                        for single_plot in tests_to_plot_together]

        z_m = fix_results['z(m)']
        visualization.plot.plot_evaluate(
            z_m, reference_values, fixed_values, acceptable_limits,
            lin_fix, 'fred', save_fig=True, num=60)

    return test_flags


def bruce_tests(lin_ref: Accelerator, lin_fix: Accelerator,
                multipart: bool = True, plot: bool = True
                ) -> dict[[str], bool]:
    """Test the fixed linac using Bruce's paper."""
    if multipart:
        ref_results = lin_ref.tracewin_simulation.results_multipart
        fix_results = lin_fix.tracewin_simulation.results_multipart
    else:
        ref_results = lin_ref.tracewin_simulation.results_envelope
        fix_results = lin_fix.tracewin_simulation.results_envelope

    fixed_values = []

    test_flags = {'relative_var_et': None,
                  'relative_var_ep': None,
                  'mismatch_t': None,
                  'mismatch_zdp': None,
                  'max_retuned_power': None}

    base = 'relative_var_'
    relative_var_of_eps = {}
    for key in ['et', 'ep']:
        delta = 100. * (fix_results[key] - ref_results[key]) / ref_results[key]
        fix_results[base + key] = delta
        test_flags[base + key] = delta[-1]
        relative_var_of_eps[base + key] = delta
    fixed_values.append(relative_var_of_eps)

    # Mismatch test
    mismatch = {'x': None, 'y': None, 'zdp': None}
    for key in mismatch.keys():
        twiss_ref = ref_results['twiss_' + key]
        twiss_fix = fix_results['twiss_' + key]
        mismatch[key] = mismatch_factor(twiss_ref, twiss_fix, transp=True)

    fix_results['mismatch_t'] = .5 * (mismatch['x'] + mismatch['y'])
    fix_results['mismatch_zdp'] = mismatch['zdp']
    mismatches = {}
    for key in ['mismatch_t', 'mismatch_zdp']:
        test_flags[key] = fix_results[key][-1]
        mismatches[key] = fix_results[key]
    fixed_values.append(mismatches)

    if plot:
        z_m = fix_results['z(m)']

        reference_values = []
        acceptable_limits = []
        for dic in fixed_values:
            tmp1, tmp2 = {}, {}
            for key, val in dic.items():
                tmp1[key] = val * np.NaN
                tmp2[key] = {'max': None, 'min': None}
            reference_values.append(tmp1)
            acceptable_limits.append(tmp2)
        visualization.plot.plot_evaluate(
            z_m, reference_values, fixed_values, acceptable_limits,
            lin_fix, 'bruce', save_fig=True, num=70)

    return test_flags
