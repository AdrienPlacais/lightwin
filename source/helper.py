#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 14:15:48 2021.

@author: placais
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
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
        else:
            yield key

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


def printd(message, color_header='cyan', header=''):
    """Print delimited message."""
    pd.options.display.float_format = '{:.4e}'.format
    pd.options.display.max_columns = 10
    pd.options.display.max_colwidth = 65
    pd.options.display.width = 250

    line = '=================================================================='
    print(line, '\n')
    if len(header) > 0:
        printc(header, color_header)
    print(message, '\n\n' + line, '\n')


# =============================================================================
# Plot functions
# =============================================================================
def simple_plot(dat_x, dat_y, label_x, label_y, fignum=33):
    """Simplest plot."""
    axnumlist = [111]
    _, axlist = create_fig_if_not_exist(fignum, axnumlist)
    axlist[0].plot(dat_x, dat_y)
    axlist[0].set_xlabel(label_x)
    axlist[0].set_ylabel(label_y)
    axlist[0].grid(True)


def create_fig_if_not_exist(fignum, axnum, sharex=False, **fkwargs):
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
        fig = plt.figure(fignum, **fkwargs)
        axlist.append(fig.add_subplot(axnum[0]))
        dict_sharex = {True: axlist[0], False: None}
        for i in axnum[1:]:
            axlist.append(fig.add_subplot(i, sharex=dict_sharex[sharex]))

    return fig, axlist


def clean_fig(fignumlist):
    """Clean axis of Figs in fignumlist."""
    for fignum in fignumlist:
        fig = plt.figure(fignum)
        for axx in fig.get_axes():
            axx.cla()


def empty_fig(fignum):
    """Return True if at least one axis of Fig(fignum) has no line."""
    out = False
    if plt.fignum_exists(fignum):
        fig = plt.figure(fignum)
        axlist = fig.get_axes()
        for axx in axlist:
            if axx.lines == []:
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
    dict_x_axis = {  # first element is patch dimension. second is x limits
        's': lambda elt, i: [
            {'x0': elt.pos_m['abs'][0], 'width': elt.length_m},
            [linac.elements['list'][0].pos_m['abs'][0],
             linac.elements['list'][-1].pos_m['abs'][-1]]
        ],
        'elt': lambda elt, i: [
            {'x0': i, 'width': 1},
            [0, i]
        ]
    }

    for i, elt in enumerate(linac.elements['list']):
        kwargs = dict_x_axis[x_axis](elt, i)[0]
        ax.add_patch(dict_elem_plot[elt.get('nature')](elt, **kwargs))

    ax.set_xlim(dict_x_axis[x_axis](elt, i)[1])
    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_ylim([-.05, 1.05])


def _plot_drift(drift, x0, width):
    """Add a little rectangle to show a drift."""
    height = .4
    y0 = .3
    patch = pat.Rectangle((x0, y0), width, height, fill=False, lw=0.5)
    return patch


def _plot_quad(quad, x0, width):
    """Add a crossed large rectangle to show a quad."""
    height = 1.
    y0 = 0.
    path = np.array(([x0, y0], [x0 + width, y0], [x0 + width, y0 + height],
                     [x0, y0 + height], [x0, y0], [x0 + width, y0 + height],
                     [np.NaN, np.NaN], [x0, y0 + height], [x0 + width, y0]))
    patch = pat.Polygon(path, closed=False, fill=False, lw=0.5)
    return patch


def _plot_field_map(field_map, x0, width):
    """Add an ellipse to show a field_map."""
    height = 1.
    y0 = height * .5
    dict_colors = {
        'nominal': 'green',
        'rephased (in progress)': 'yellow',
        'rephased (ok)': 'yellow',
        'failed': 'red',
        'compensate (in progress)': 'orange',
        'compensate (ok)': 'orange',
        'compensate (not ok)': 'orange',
    }
    patch = pat.Ellipse((x0 + .5 * width, y0), width, height, fill=True,
                        lw=0.5, fc=dict_colors[field_map.get('status')],
                        ec='k')
    return patch


def plot_section(linac, ax, x_axis='s'):
    """Add light grey rectangles behind the plot to show the sections."""
    dict_x_axis = {
        'last_elt_of_sec': lambda sec: sec[-1][-1],
        's': lambda elt: linac.synch.z['abs_array'][elt.idx['s_out']],
        'elt': lambda elt: elt.idx['element'] + 1,
    }
    x_ax = [0]
    for i, section in enumerate(linac.elements['l_sections']):
        elt = dict_x_axis['last_elt_of_sec'](section)
        x_ax.append(dict_x_axis[x_axis](elt))

    for i in range(len(x_ax) - 1):
        if i % 2 == 1:
            ax.axvspan(x_ax[i], x_ax[i + 1], ymin=-1e8, ymax=1e8, fill=True,
                       alpha=.1, fc='k')


def _compute_ellipse_parameters(d_eq):
    """
    Compute the ellipse parameters so as to plot the ellipse.

    Parameters
    ----------
    d_eq : dict
        Holds ellipe equations parameters, defined as:
            Ax**2 + Bxy + Cy**2 + Dx + Ey + F = 0

    Return
    ------
    d_plot : dict
        Holds semi axis, center of ellipse, angle.
    """
    delta = d_eq["B"]**2 - 4. * d_eq["A"] * d_eq["C"]
    tmp1 = d_eq["A"] * d_eq["E"]**2 - d_eq["C"] * d_eq["D"]**2 \
        - d_eq["B"] * d_eq["D"] * d_eq["E"] + delta * d_eq["F"]
    tmp2 = np.sqrt((d_eq["A"] - d_eq["C"])**2 + d_eq["B"]**2)

    if np.abs(d_eq["B"]) < 1e-8:
        if d_eq["A"] < d_eq["C"]:
            theta = 0.
        else:
            theta = np.pi/2.
    else:
        theta = np.arctan((d_eq["C"] - d_eq["A"] - tmp2) / d_eq["B"])

    d_plot = {
        "a": -np.sqrt(2. * tmp1 * (d_eq["A"] + d_eq["C"] + tmp2)) / delta,
        "b": -np.sqrt(2. * tmp1 * (d_eq["A"] + d_eq["C"] - tmp2)) / delta,
        "x0": (2. * d_eq["C"] * d_eq["D"] - d_eq["B"] * d_eq["E"]) / delta,
        "y0": (2. * d_eq["A"] * d_eq["E"] - d_eq["B"] * d_eq["D"]) / delta,
        "theta": theta,
    }
    return d_plot


def plot_ellipse(axx, d_eq, **plot_kwargs):
    """The proper ellipse plotting."""
    d_plot = _compute_ellipse_parameters(d_eq)
    n_points = 10001
    var = np.linspace(0., 2. * np.pi, n_points)
    ellipse = np.array([d_plot["a"] * np.cos(var), d_plot["b"] * np.sin(var)])
    rotation = np.array([[np.cos(d_plot["theta"]), -np.sin(d_plot["theta"])],
                         [np.sin(d_plot["theta"]),  np.cos(d_plot["theta"])]])
    ellipse_rot = np.empty((2, n_points))

    for i in range(n_points):
        ellipse_rot[:, i] = np.dot(rotation, ellipse[:, i])

    axx.plot(d_plot["x0"] + ellipse_rot[0, :],
             d_plot["y0"] + ellipse_rot[1, :],
             lw=0., marker='o', ms=.5, **plot_kwargs)


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
    n_z = lin.synch.z['abs_array'].shape[0]
    data = np.column_stack((
        lin.synch.z['abs_array'],
        lin.synch.energy['kin_array_mev'],
        lin.synch.phi['abs_array'],
        np.reshape(lin.transf_mat['cumul'], (n_z, 4))
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
