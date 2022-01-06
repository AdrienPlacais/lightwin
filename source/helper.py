#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
import constants


# =============================================================================
# Plot and messages functions
# =============================================================================
def printc(message, color='red', opt_message=''):
    """Print colored messages."""
    if color in ('red', 'r', 'warning'):
        escape_code = '\x1B[31m'

    if color in ('blue', 'b', 'message'):
        escape_code = '\x1b[34m'

    if color in ('green', 'g', 'results'):
        escape_code = '\x1b[32m'

    if color in ('magenta', 'm', 'error'):
        escape_code = '\x1b[35m'

    if color in ('cyan', 'c', 'info'):
        escape_code = '\x1b[36m'

    normal_code = '\x1b[0m'

    print(escape_code + message + normal_code + opt_message)


def simple_plot(x, y, label_x, label_y, fignum=33):
    """Simplest plot."""
    axnumlist = [111]
    fig, axlist = create_fig_if_not_exist(fignum, axnumlist)
    axlist[0].plot(x, y)
    axlist[0].set_xlabel(label_x)
    axlist[0].set_ylabel(label_y)
    axlist[0].grid(True)


def create_fig_if_not_exist(fignum, axnum):
    """
    Check if figures were already created, create it if not.

    Parameters
    ----------
    fignum: int
        Number of the fignum.
    axnum: list of int
        Axes indexes as understood by fig.add_subplot
    """
    n_axes = len(axnum)
    axlist = []

    if plt.fignum_exists(fignum):
        fig = plt.figure(fignum)
        for i in range(n_axes):
            axlist.append(fig.axes[i])

    else:
        fig = plt.figure(fignum)
        for i in axnum:
            axlist.append(fig.add_subplot(i))

    return fig, axlist


def clean_fig(fignumlist):
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        for ax in fig.get_axes():
            ax.cla()


def empty_fig(fignum):
    """Return True if at least one axis of Fig(fignum) has no line."""
    out = False
    if plt.fignum_exists(fignum):
        fig = plt.figure(fignum)
        axlist = fig.get_axes()
        for ax in axlist:
            if ax.lines == []:
                out = True
    return out


def plot_pty_with_data_tags(ax, x, y, idx_list, tags=True):
    """
    Plot y vs x.

    Data at idx_list are magnified with bigger points and data tags.
    """
    line, = ax.plot(x, y)
    ax.scatter(x[idx_list], y[idx_list], color=line.get_color())

    if tags:
        n = len(idx_list)
        for i in range(n):
            txt = str(np.round(x[idx_list][i], 4)) + ',' \
                + str(np.round(y[idx_list][i], 4))
            ax.annotate(txt,
                        (x[idx_list][i], y[idx_list[i]]),
                        size=8)


# =============================================================================
# Files functions
# =============================================================================
def save_full_mt_and_energy_evolution(accelerator):
    """
    Output the energy and transfer matrice components as a function of z.

    z [m]   E[MeV]  M_11    M_12    M_21    M_22

    Parameters
    ----------
    accelerator: Accelerator object
        Object of corresponding to desired output.
    """
    data = accelerator.full_MT_and_energy_evolution
    n_z = data.shape[0]
    filepath = '../data/full_energy_and_MT.txt'
    out = np.full((n_z, 6), np.NaN)

    for i in range(n_z):
        out[i, :] = data[i, :, :].flatten()

    np.savetxt(filepath, out)


def save_vcav_and_phis(accelerator):
    """
    Output the Vcav and phi_s as a function of z.

    z [m]   V_cav[MV]  phi_s[deg]

    Parameters
    ----------
    accelerator: Accelerator object
        Object of corresponding to desired output.
    """
    data_v = np.copy(accelerator.V_cav_MV)
    valid_idx = np.where(~np.isnan(data_v))
    data_v = data_v[valid_idx]
    data_phi_s = np.copy(accelerator.phi_s_deg)[valid_idx]
    data_z = np.copy(accelerator.absolute_entrance_position)[valid_idx] \
        + np.copy(accelerator.L_m)[valid_idx]

    out = np.transpose(np.vstack((data_z, data_v, data_phi_s)))
    filepath = '../data/Vcav_and_phis.txt'

    np.savetxt(filepath, out)


# =============================================================================
# Conversion functions
# =============================================================================
def v_to_mev(v, m_over_q):
    """Convert m/s to MeV."""
    e_ev = 0.5 * m_over_q * v**2
    return e_ev * 1e-6


def mev_to_v(e_mev, q_over_m):
    """Convert MeV to m/s."""
    v = np.sqrt(2e6 * q_over_m * e_mev)
    return v


def mev_to_gamma(energy_mev, mass_mev):
    """Convert MeV energy into Lorentz gamma."""
    gamma = 1. + energy_mev / mass_mev
    return gamma


def gamma_to_mev(gamma, mass_mev):
    """Convert MeV energy into Lorentz gamma."""
    energy_mev = mass_mev * (gamma - 1.)
    return energy_mev


def gamma_to_beta(gamma):
    """Convert MeV energy into Lorentz beta."""
    beta = np.sqrt(1. - gamma**-2)
    return beta


def mev_to_p(energy_mev, mass_mev, q_over_m, mass_kg):
    """Convert MeV energy to p."""
    v = mev_to_v(energy_mev, q_over_m)
    gamma = mev_to_gamma(energy_mev, mass_mev)
    return gamma * mass_kg * v


def mev_and_gamma_to_p(energy_mev, gamma, mass_kg, q_over_m):
    """Compute p when energy and gamma are already calculated."""
    return gamma * mass_kg * mev_to_v(energy_mev, q_over_m)


def gamma_and_beta_to_p(gamma, beta, mass_kg):
    """Compute p when gamma and beta are already known."""
    return gamma * beta * constants.c * constants.m_kg


def phi_to_z(phi, beta, omega):
    """Return the distance crossed during phi at speed beta."""
    return -beta * constants.c * phi / omega


def z_to_phi(z, beta, omega):
    """Convert (delta) position into (delta) phase."""
    return -omega * z / (beta * constants.c)


def mrad_and_gamma_to_delta(z_prime, gamma):
    """Convert z' in mrad with gamma to delta = dp/p in %."""
    return z_prime * gamma**2 * 1e-1


def mrad_and_mev_to_delta(z_prime, e_mev, mass_mev):
    """Convert z' in mrad with energy to delta = dp/p in %."""
    gamma = mev_to_gamma(e_mev, mass_mev)
    return mrad_and_gamma_to_delta(z_prime, gamma)


# =============================================================================
# Matrix manipulation
# =============================================================================
def individual_to_global_transfer_matrix(m_in):
    """
    Compute the transfer matrix of several elements.

    Parameters
    ----------
    m_in: dim 3 np.array
        Array of the form (n, 2, 2). Transfer matrices of INDIVIDUAL elements.

    Return
    ------
    m_out: dim 3 np.array
        Same shape as M. Contains transfer matrices of line from the start of
        the line.
    """
    m_out = np.full_like(m_in, np.NaN)
    m_out[0, :, :] = m_in[0, :, :]

    n = m_in.shape[0]
    for i in range(1, n):
        m_out[i, :, :] = m_in[i, :, :] @ m_out[i-1, :, :]

    return m_out


def right_recursive_matrix_product(m_in, idx_min, idx_max):
    """
    Compute the matrix product along the last array. For transfer matrices.

    Parameters
    ----------
    M: dim 3 np.array
        Array of the form (2, 2, n).
    idx_min: int
        First index to consider.
    idx_max: int
        Last index to consider.
    """
    m_out = np.eye(2)

    for i in range(idx_min, idx_max + 1):
        m_out = m_in[i, :, :] @ m_out

    return m_out
