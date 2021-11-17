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
def load_electric_field_1D(path):
    """
    Load a 1D electric field (.edz extension).

    Parameters
    ----------
    path: string
        The path to the .edz file to load.

    Returns
    -------
    Fz: np.array
        Array of electric field in MV/m.
    zmax: float
        z position of the filemap end.
    Norm: float
        Norm of the electric field.

    Currently not returned
    ----------------------
    nz: int
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

                nz = int(line_splitted[0])
                # Sometimes there are several spaces or tabs between numbers
                zmax = float(line_splitted[-1])
                Fz = np.full((nz + 1), np.NaN)

            elif i == 1:
                Norm = float(line)

            else:
                Fz[k] = float(line)
                k += 1

            i += 1

    return nz, zmax, Norm, Fz


def save_full_MT_and_energy_evolution(accelerator):
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


def save_Vcav_and_phis(accelerator):
    """
    Output the Vcav and phi_s as a function of z.

    z [m]   V_cav[MV]  phi_s[deg]

    Parameters
    ----------
    accelerator: Accelerator object
        Object of corresponding to desired output.
    """
    data_V = np.copy(accelerator.V_cav_MV)
    valid_idx = np.where(~np.isnan(data_V))
    data_V = data_V[valid_idx]
    data_phi_s = np.copy(accelerator.phi_s_deg)[valid_idx]
    data_z = np.copy(accelerator.absolute_entrance_position)[valid_idx] \
        + np.copy(accelerator.L_m)[valid_idx]

    out = np.transpose(np.vstack((data_z, data_V, data_phi_s)))
    filepath = '../data/Vcav_and_phis.txt'

    np.savetxt(filepath, out)


# =============================================================================
# Conversion functions
# =============================================================================
def v_to_MeV(v, m_over_q):
    """Convert m/s to MeV."""
    E_eV = 0.5 * m_over_q * v**2
    return E_eV * 1e-6


def MeV_to_v(E_MeV, q_over_m):
    """Convert MeV to m/s."""
    v = np.sqrt(2. * q_over_m * 1e6 * E_MeV)
    return v

def mev_to_gamma(energy_mev, mass_mev):
    """Convert MeV energy into Lorentz gamma."""
    gamma = 1. + energy_mev / mass_mev
    return gamma


# =============================================================================
# Matrix manipulation
# =============================================================================
def individual_to_global_transfer_matrix(M):
    """
    Compute the transfer matrix of several elements.

    Parameters
    ----------
    M: dim 3 np.array
        Array of the form (n, 2, 2). Transfer matrices of INDIVIDUAL elements.

    Return
    ------
    M_tot: dim 3 np.array
        Same shape as M. Contains transfer matrices of line from the start of
        the line.
    """
    M_tot = np.full_like(M, np.NaN)
    M_tot[0, :, :] = M[:, :, 0]

    n = M.shape[0]
    for i in range(1, n):
        M_tot[i, :, :] = M[i, :, :] @ M_tot[i-1, :, :]

    return M_tot


def right_recursive_matrix_product(M, idx_min, idx_max):
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
    M_out = np.eye(2)

    for i in range(idx_min, idx_max + 1):
        M_out = M[:, :, i] @ M_out

    return M_out


# =============================================================================
# Integration
# =============================================================================
def RK4(u, du_dx, x, dx):
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

