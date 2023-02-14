#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 13:52:26 2023.

@author: placais

Routines to evaluate the quality of the new settings for the linac.
"""
import numpy as np
import matplotlib.pyplot as plt

from palettable.colorbrewer.qualitative import Dark2_8
from cycler import cycler

from util import helper
from core.emittance import mismatch_factor

font = {'family': 'serif',
        'size': 20}
plt.rc('font', **font)
plt.rc('axes', prop_cycle=(cycler('color', Dark2_8.mpl_colors)))
plt.rc('mathtext', fontset='cm')


def multipart_flags_test(lin_ref, lin_fix, multipart=True, plot=True):
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
    source = "multipart"
    if not multipart:
        source = "envelope"
    d_ref = lin_ref.tw_results[source]
    d_fix = lin_fix.tw_results[source]

    d_valid = {'Powlost': True, 'e': True, 'e99': True}
    z_m = d_fix['z(m)']

    # Power loss test
    pow_lost = d_fix['Powlost']
    if pow_lost[-1] > 1e-10:
        print("Loss of power!")
        d_valid['Powlost'] = False

    # RMS emittances test
    eps_rms = np.column_stack((d_fix['ex'], d_fix['ey'], d_fix['ep']))
    var_rms = 100. * (eps_rms - eps_rms[0, :]) / eps_rms[0, :]
    if np.any(np.where(var_rms > 20.)):
        print("The RMS emittance is too damn high!")
        d_valid['e'] = False

    # 99% emittances test
    eps99 = np.column_stack((d_fix['ex99'], d_fix['ey99'], d_fix['ep99']))
    eps99_ref = np.column_stack((d_ref['ex99'], d_ref['ey99'], d_ref['ep99']))

    var_max_99 = 100. * (np.max(eps99, axis=0) - np.max(eps99_ref, axis=0)) \
        / np.max(eps99, axis=0)
    if np.any(var_max_99 > 30.):
        print("The 99% emittance is too damn high!")
        d_valid['e99'] = False

    if plot:
        _plot_multipart_flags_test(z_m, pow_lost, var_rms, eps99, eps99_ref)

    return d_valid


def _plot_multipart_flags_test(z_m, pow_lost, var_rms, eps_99, eps_99_ref):
    """Plot quantities and their limits for flags test."""
    fig, axx = plt.subplots(3, 1)
    axx[0].set_ylabel('Lost power [%]')
    axx[1].set_ylabel(r'$\Delta\epsilon_{RMS}/\epsilon_{RMS}^{z_0}$ [%]')
    axx[2].set_ylabel(r'$\epsilon_{99}$')
    axx[-1].set_xlabel('Position [m]')

    axx[0].plot(z_m, pow_lost)

    lab = ['ex', 'ey', 'ep']
    for i in range(3):
        axx[1].plot(z_m, var_rms[:, i], label=lab[i])
    axx[1].axhline(20, xmin=z_m[0], xmax=z_m[-1], c='r', lw=4)
    axx[1].legend()

    for i in range(3):
        line, = axx[2].plot(z_m, eps_99_ref[:, i], label='ref', ls='--')
        axx[2].plot(z_m, eps_99[:, i], label='fix', color=line.get_color())
        axx[2].axhline(1.3 * np.max(eps_99_ref[:, i]),
                       xmin=z_m[0], xmax=z_m[-1], lw=4, color=line.get_color())
        if i == 0:
            axx[2].legend()

    for i in range(3):
        axx[i].grid(True)


def bruce_tests(lin_ref, lin_fix, multipart=True, plot=False):
    """Test the fixed linac using Bruce's paper."""
    source = "multipart"
    if not multipart:
        source = "envelope"

    d_ref = lin_ref.tw_results[source]
    d_fix = lin_fix.tw_results[source]

    d_tests = {'var_eps_transv': None,
               'var_eps_long': None,
               'mismatch_transv': None,
               'mismatch_long': None,
               'max_retuned_power': None}

    z_m = d_fix['z(m)']

    # Emittances
    eps_ref = {'transv': (d_ref['ex'] + d_ref['ey']) / 2.,
               'long': d_ref['ep']}
    eps_fix = {'transv': (d_fix['ex'] + d_fix['ey']) / 2.,
               'long': d_fix['ep']}
    delta = {}

    for key in ['transv', 'long']:
        delta[key] = 100. * (eps_fix[key] - eps_ref[key]) / eps_ref[key]
        d_tests['var_eps_' + key] = delta[key][-1]

    # Mismatch test
    d_planes = {"xx'": ["SizeX", "sxx'", "ex"],
                "yy'": ["SizeY", "syy'", "ey"],
                "zdp": ["SizeZ", "szdp", "ezdp"]}

    twiss = {}
    mismatch = {}
    for key, val in d_planes.items():
        for lin, name in zip([d_fix, d_ref], ['fix', 'ref']):
            # We need to unnormalize emittances
            eps = lin[val[2]].copy()
            gamma_lorentz = lin['gama-1'] + 1.
            beta_lorentz = lin['beta']
            eps /= (beta_lorentz * gamma_lorentz)

            # Here, alpha beta and gamma are Twiss
            alpha = -lin[val[1]] / eps
            beta = lin[val[0]]**2 / eps
            if key == 'zdp':
                beta *= 10.
            gamma = (1. + alpha**2) / beta

            twiss[key + name] = np.column_stack((alpha, beta, gamma))

        mismatch[key] = mismatch_factor(twiss[key + 'ref'], twiss[key + 'fix'],
                                        transp=True)
    d_tests["mismatch_transv"] = (mismatch["xx'"][-1] + mismatch["yy'"][-1]) \
        / 2.
    d_tests["mismatch_long"] = mismatch["zdp"][-1]

    if plot:
        _plot_bruce_tests(z_m, delta, mismatch)

    return d_tests


def _plot_bruce_tests(z_m, delta_eps, mismatch):
    """Output what is calculated."""
    fig = plt.figure(50)
    ax = fig.add_subplot(111)
    ax.set_xlabel("z [m]")
    ax.set_ylabel(r"$\Delta\epsilon/\epsilon_0$ (RMS)")
    ax.plot(z_m, delta_eps["transv"], label="Transverse")
    line, = ax.plot(z_m, delta_eps["long"], label="Longitudinal")

    ax.legend()
    ax.grid(True)

    fig = plt.figure(51)
    ax = fig.add_subplot(111)
    ax.set_xlabel("z [m]")
    ax.set_ylabel(r"$M$")
    ax.plot(z_m, mismatch["xx'"], label="xx'")
    ax.plot(z_m, mismatch["yy'"], label="yy'")
    line1, = ax.plot(z_m, mismatch["zdp"], label="zdp")

    ax.legend()
    ax.grid(True)
