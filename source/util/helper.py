#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021.

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from constants import c, E_REST_MEV


# =============================================================================
# Misc
# =============================================================================
def recursive_items(dictionary):
    """Recursively list all keys of a possibly nested dictionary."""
    for key, value in dictionary.items():
        if isinstance(value, dict):
            yield key
            yield from recursive_items(value)
        elif hasattr(value, 'has'):
            yield key
            yield from recursive_items(vars(value))
            # for ListOfElements:
            if isinstance(value, list):
                yield from recursive_items(vars(value[0]))
        else:
            yield key


def recursive_getter(key, dictionary, **kwargs):
    """Get first key in a possibly nested dictionary."""
    for _key, _value in dictionary.items():
        if key == _key:
            return _value

        if isinstance(_value, dict):
            value = recursive_getter(key, _value, **kwargs)
            if value is not None:
                return value

        elif hasattr(_value, 'get'):
            value = _value.get(key, **kwargs)
            if value is not None:
                return value
    return None


# =============================================================================
# Messages functions
# =============================================================================
# TODO: transform inputs into strings if they are not already strings
# TODO: use args to avoid lenghty 'opt_message=' each time
def printc(message, color='cyan', opt_message=''):
    """Print colored messages."""
    dict_c = {
        'red': '\x1B[31m',
        'blue': '\x1b[34m',
        'green': '\x1b[32m',
        'magenta': '\x1b[35m',
        'cyan': '\x1b[36m',
        'normal': '\x1b[0m',
    }
    print(dict_c[color] + message + dict_c['normal'] + opt_message)


# TODO: replace nan by ' ' when there is a \n in a pd DataFrame header
def printd(message, color_header='cyan', header=''):
    """Print delimited message."""
    pd.options.display.float_format = '{:.6f}'.format
    pd.options.display.max_columns = 10
    pd.options.display.max_colwidth = 18
    pd.options.display.width = 250

    tot_width = 100

    print('\n' + '=' * tot_width)
    if len(header) > 0:
        printc(header, color_header)

    # Output multi-line for headers
    if isinstance(message, pd.DataFrame):
        message.columns = message.columns.str.split("\n", expand=True)
    print(message, '\n' + '=' * tot_width, '\n')


def resample(x_1, y_1, x_2, y_2):
    """Downsample y_highres(olution) to x_1 or x_2 (the one with low res)."""
    assert x_1.shape == y_1.shape
    assert x_2.shape == y_2.shape

    if x_1.shape > x_2.shape:
        y_1 = np.interp(x_2, x_1, y_1)
        x_1 = x_2
    else:
        y_2 = np.interp(x_1, x_2, y_2)
        x_2 = x_1
    return x_1, y_1, x_2, y_2


# =============================================================================
# Files functions
# =============================================================================
def save_energy_phase_tm(lin):
    """
    Save energy, phase, transfer matrix as a function of s.

    s [m]   E[MeV]  phi[rad]  M_11    M_12    M_21    M_22

    Parameters
    ----------
    lin : Accelerator object
        Object of corresponding to desired output.
    """
    n_z = lin.get('z_abs').shape[0]
    data = np.column_stack((lin.get('z_abs'), lin.get('w_kin'),
                            lin.get('phi_abs_array'),
        np.reshape(lin.transf_mat['tm_cumul'], (n_z, 4))
    ))
    filepath = lin.files['results_folder'] + lin.name \
        + '_energy_phase_tm.txt'
    filepath = filepath.replace(' ', '_')
    header = 's [m] \t W_kin [MeV] \t phi_abs [rad]' \
        + '\t M_11 \t M_12 \t M_21 \t M_22'
    np.savetxt(filepath, data, header=header)
    print(f"Energy, phase and TM saved in {filepath}")


def save_vcav_and_phis(lin):
    """
    Output the Vcav and phi_s as a function of z.

    s [m]   V_cav [MV]  phi_s [deg]

    Parameters
    ----------
    accelerator: Accelerator object
        Object of corresponding to desired output.
    """
    printc("helper.save_vcav_and_phis warning: ", opt_message="s [m] not "
           + "saved.")
    # data = lin.get('abs', 'v_cav_mv', 'phi_s', to_deg=True)
    data = lin.get('v_cav_mv', 'phi_s', to_deg=True)
    data = np.column_stack((data[0], data[1]))

    filepath = lin.files['results_folder'] + lin.name + '_Vcav_and_phis.txt'
    filepath = filepath.replace(' ', '_')

    header = 's [m] \t V_cav [MV] \t phi_s [deg]'
    np.savetxt(filepath, data, header=header)
    print(f"Cavities accelerating field and synch. phase saved in {filepath}")


# =============================================================================
# Conversion functions
# =============================================================================
def v_to_kin_mev(v, m_over_q):
    """Convert velocity in m/s to kinetic energy in MeV."""
    e_kin_ev = 0.5 * m_over_q * v**2
    return e_kin_ev * 1e-6


def kin_mev_to_v(e_kin_mev, q_over_m):
    """Convert kinetic energy in MeV to m/s velocity."""
    v = np.sqrt(2e6 * q_over_m * e_kin_mev)
    return v


def kin_to_gamma(e_kin, e_rest=E_REST_MEV):
    """Convert kinetic and rest energies into relativistic mass factor."""
    gamma = 1. + e_kin / e_rest
    return gamma    # Both energies in same unit


def gamma_to_kin(gamma, e_rest=E_REST_MEV):
    """Convert relativistic mass factor and rest energy to kinetic energy."""
    e_kin = e_rest * (gamma - 1.)
    return e_kin    # Same unit as e_rest


def gamma_to_beta(gamma):
    """Convert relativistic mass factor to relativistic velocity factor."""
    beta = np.sqrt(1. - gamma**-2)
    return beta


def kin_to_beta(e_kin, e_rest=E_REST_MEV):
    """Convert kinetic and rest energies into reduced velocity."""
    gamma = kin_to_gamma(e_kin, e_rest)
    beta = gamma_to_beta(gamma)
    return beta     # Same unit for both energies


def kin_to_p(e_kin, e_rest=E_REST_MEV):
    """Convert kinetic energy to impulsion."""
    e_tot = e_kin + e_rest
    p = np.sqrt(e_tot**2 - e_rest**2)
    return p    # If energies in MeV, p in MeV/c


def p_to_kin(p, e_rest=E_REST_MEV):
    """Convert impulsion to kinetic energy."""
    e_tot = np.sqrt(p**2 + e_rest**2)
    e_kin = e_tot - e_rest
    return e_kin    # Attention to units!


def gamma_and_beta_to_p(gamma, beta, e_rest=E_REST_MEV):
    """Compute p from Lorentz factors."""
    return gamma * beta * e_rest


def phi_to_z(phi, beta, omega):
    """Return the distance crossed during phi at speed beta."""
    return -beta * c * phi / omega


def z_to_phi(z, beta, omega):
    """Convert (delta) position into (delta) phase."""
    return -omega * z / (beta * c)


def mrad_and_gamma_to_delta(z_prime, gamma):
    """Convert z' in mrad with gamma to delta = dp/p in %."""
    return z_prime * gamma**2 * 1e-1


def mrad_and_mev_to_delta(z_prime, e_kin, e_rest=E_REST_MEV):
    """Convert z' in mrad with energy to delta = dp/p in %."""
    gamma = kin_to_gamma(e_kin, e_rest)
    return mrad_and_gamma_to_delta(z_prime, gamma)


def diff_angle(phi_1, phi_2):
    """Compute smallest difference between two angles."""
    delta_phi = np.arctan2(
        np.sin(phi_2 - phi_1),
        np.cos(phi_2 - phi_1)
    )
    return delta_phi


# =============================================================================
# Matrix manipulation
# =============================================================================
def individual_to_global_transfer_matrix(m_in, m_out, idxs=None):
    """
    Compute the transfer matrix of several elements.

    For efficiency reasons, we compute transfer matrices only between idxs[0]
    and idxs[1]. If idxs is not provided, or if it matches the full dimensions
    of the linac, we recompute the full linac.

    Parameters
    ----------
    m_in : np.array
        Array of the form (n, 2, 2). Transfer matrices of INDIVIDUAL elements.
    m_out : np.array
        Array of the form (n, 2, 2). Transfer matrices of from the start of the
        line.
    idxs : list
        First and last index of the matrix to recompute.

    Return
    ------
    m_out : np.array
        Array of the form (n, 2, 2). Transfer matrices of from the start of the
        line.
    """
    if idxs is None:
        idxs = [0, m_in.shape[0]]

    if idxs == [0, m_in.shape[0]]:
        m_out[0, :, :] = np.eye(2)

    for i in range(idxs[0] + 1, idxs[1]):
        m_out[i, :, :] = m_in[i, :, :] @ m_out[i - 1, :, :]
