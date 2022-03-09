#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import constants


# =============================================================================
# Plot and messages functions
# =============================================================================
def printc(message, color='cyan', opt_message=''):
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


def printd(message, color_header='cyan', header=''):
    """Print delimited message."""
    line = '=================================================================='
    print(line, '\n')
    if len(header) > 0:
        printc(header, color_header)
        print('\n')
    print(message, '\n', line, '\n')


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


def plot_structure(linac, ax, x_axis='s'):
    """Plot a structure of the linac under study."""
    dict_elem_plot = {
        'DRIFT': _plot_drift,
        'QUAD': _plot_quad,
        'FIELD_MAP': _plot_field_map,
        }
    i = 0
    for elt in linac.elements['list']:
        if x_axis == 's':
            x0 = elt.pos_m['abs'][0]
            width = elt.length_m
        elif x_axis == 'elt':
            x0 = i
            width = 1
        ax.add_patch(dict_elem_plot[elt.info['name']](elt, x0, width))
        i += 1

    if x_axis == 's':
        ax.set_xlim([linac.elements['list'][0].pos_m['abs'][0],
                     linac.elements['list'][-1].pos_m['abs'][-1]])
    elif x_axis == 'elt':
        ax.set_xlim([0, i])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylim([-.05, 1.05])


def _plot_drift(drift, x0, width):
    """Add a little rectangle to show a drift."""
    height = .4
    y0 = .3
    patch = pat.Rectangle((x0, y0), width, height, fill=False)
    return patch


def _plot_quad(quad, x0, width):
    """Add a crossed large rectangle to show a quad."""
    height = 1.
    y0 = 0.
    path = np.array(([x0, y0], [x0 + width, y0], [x0 + width, y0 + height],
                     [x0, y0 + height], [x0, y0], [x0 + width, y0 + height],
                     [np.NaN, np.NaN], [x0, y0 + height], [x0 + width, y0]))
    patch = pat.Polygon(path, closed=False, fill=False)
    return patch


def _plot_field_map(field_map, x0, width):
    """Add an ellipse to show a field_map."""
    height = 1.
    y0 = height * .5
    if field_map.info['failed']:
        color = 'red'
    else:
        if field_map.info['compensate']:
            color = 'orange'
        else:
            color = 'green'
    patch = pat.Ellipse((x0 + .5*width, y0), width, height, fill=True,
                        fc=color, ec='k')
    return patch


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
def v_to_kin_mev(v, m_over_q):
    """Convert velocity in m/s to kinetic energy in MeV."""
    e_kin_ev = 0.5 * m_over_q * v**2
    return e_kin_ev * 1e-6


def kin_mev_to_v(e_kin_mev, q_over_m):
    """Convert kinetic energy in MeV to m/s velocity."""
    v = np.sqrt(2e6 * q_over_m * e_kin_mev)
    return v


def kin_to_gamma(e_kin, e_rest):
    """Convert kinetic and rest energies into relativistic mass factor."""
    gamma = 1. + e_kin / e_rest
    return gamma    # Both energies in same unit


def gamma_to_kin(gamma, e_rest):
    """Convert relativistic mass factor and rest energy to kinetic energy."""
    e_kin = e_rest * (gamma - 1.)
    return e_kin    # Same unit as e_rest


def gamma_to_beta(gamma):
    """Convert relativistic mass factor to relativistic velocity factor."""
    beta = np.sqrt(1. - gamma**-2)
    return beta


def kin_to_beta(e_kin, e_rest):
    """Convert kinetic and rest energies into reduced velocity."""
    gamma = kin_to_gamma(e_kin, e_rest)
    beta = gamma_to_beta(gamma)
    return beta     # Same unit for both energies


def kin_to_p(e_kin, e_rest):
    """Convert kinetic energy to impulsion."""
    e_tot = e_kin + e_rest
    p = np.sqrt(e_tot**2 - e_rest**2)
    return p    # If energies in MeV, p in MeV/c


def p_to_kin(p, e_rest):
    """Convert impulsion to kinetic energy."""
    e_tot = np.sqrt(p**2 + e_rest**2)
    e_kin = e_tot - e_rest
    return e_kin    # Attention to units!


def gamma_and_beta_to_p(gamma, beta, e_rest):
    """Compute p from Lorentz factors."""
    return gamma * beta * e_rest


def phi_to_z(phi, beta, omega):
    """Return the distance crossed during phi at speed beta."""
    return -beta * constants.c * phi / omega


def z_to_phi(z, beta, omega):
    """Convert (delta) position into (delta) phase."""
    return -omega * z / (beta * constants.c)


def mrad_and_gamma_to_delta(z_prime, gamma):
    """Convert z' in mrad with gamma to delta = dp/p in %."""
    return z_prime * gamma**2 * 1e-1


def mrad_and_mev_to_delta(z_prime, e_kin, e_rest):
    """Convert z' in mrad with energy to delta = dp/p in %."""
    gamma = kin_to_gamma(e_kin, e_rest)
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
