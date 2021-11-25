#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt


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

    if(plt.fignum_exists(fignum)):
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


# =============================================================================
# Files functions
# =============================================================================
def load_electric_field_1d(path):
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path: string
        The path to the .edz file to load.

    Returns
    -------
    f_z: np.array
        Array of electric field in MV/m.
    zmax: float
        z position of the filemap end.
    norm: float
        norm of the electric field.

    Currently not returned
    ----------------------
    n_z: int
        Number of points in the array minus one.
    """
    i = 0
    k = 0

    with open(path) as file:
        for line in file:
            if i == 0:
                line_splitted = line.split(' ')

                # Sometimes the separator is a tab and not a space:
                if len(line_splitted) < 2:
                    line_splitted = line.split('\t')

                n_z = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                f_z = np.full((n_z + 1), np.NaN)

            elif i == 1:
                norm = float(line)

            else:
                f_z[k] = float(line)
                k += 1

            i += 1

    return n_z, zmax, norm, f_z


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
    v = np.sqrt(2. * q_over_m * 1e6 * e_mev)
    return v


def mev_to_gamma(energy_mev, mass_mev):
    """Convert MeV energy into Lorentz gamma."""
    gamma = 1. + energy_mev / mass_mev
    return gamma


def gamma_to_mev(gamma, mass_mev):
    """Convert MeV energy into Lorentz gamma."""
    energy_mev = mass_mev * (gamma - 1.)
    return energy_mev


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


# =============================================================================
# Integration
# =============================================================================
def rk4(u, du_dx, x, dx):
    """
    4-th order Runge-Kutta integration.

    This function calculates the variation of u between x and x+dx.

    Parameters
    ----------
    u: np.array
        Holds the value of the function to integrate in x.
    df_dx: function
        Gives the variation of u components with x.
    x: real
        Where u is known.
    dx: real
        Integration step.

    Return
    ------
    delta_u: real
        Variation of u between x and x+dx.
    """
    half_dx = .5 * dx
    k_1 = du_dx(x, u)
    k_2 = du_dx(x + half_dx, u + half_dx * k_1)
    k_3 = du_dx(x + half_dx, u + half_dx * k_2)
    k_4 = du_dx(x + dx, u + dx * k_3)
    delta_u = (k_1 + 2.*k_2 + 2.*k_3 + k_4) * dx / 6.
    return delta_u
