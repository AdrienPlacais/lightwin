#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:42:41 2021.

@author: placais

Longitudinal RMS emittances:
    eps_zdelta in [z-delta]           [pi.m.rad]    non normalized
    eps_z      in [z-z']              [pi.mm.mrad]  non normalized
    eps_w      in [Delta phi-Delta W] [pi.deg.MeV]  normalized


Twiss:
    beta, gamma are Lorentz factors.
    beta_blabla, gamma_blabla are Twiss parameters.

    beta_zdelta in [z-delta]            [mm/(pi.%)]
    beta_z      in [z-z']               [mm/(pi.mrad)]
    beta_w is   in [Delta phi-Delta W]  [deg/(pi.MeV)]

    (same for gamma_z, gamma_z, gamma_zdelta)

    Conversions for alpha are easier:
        alpha_w = -alpha_z = -alpha_zdelta

TODO: handle error on eps_zdelta
TODO better ellipse plot
FIXME handle portions of linac for fit process
FIXME r_zz should be an argument instead of taking the linac attribute. Also
"""

import numpy as np
import pandas as pd
import helper
from constants import E_rest_MeV, LAMBDA_BUNCH, SIGMA_ZDELTA
import tracewin_interface as tw


# =============================================================================
# Public
# =============================================================================
def beam_parameters_zdelta(r_zz, sigma_in=SIGMA_ZDELTA):
    """
    Compute sigma beam matrix, emittance, Twiss parameters.

    Parameters
    ----------
    r_zz : numpy array
        (n, 2, 2) cumulated transfer matrices.
    sigma_in : numpy array
        (2, 2) sigma beam matrix at entry of linac.
    """
    # Compute sigma beam matrices
    sigma = _sigma_beam_matrices(r_zz, sigma_in)

    # Compute emittance and Twiss parameters in the z-delta plane.
    eps_zdelta = _emittance_zdelta(sigma)
    twiss_zdelta = _twiss_zdelta(sigma, eps_zdelta)
    enveloppes_zdelta = _enveloppes(twiss_zdelta, eps_zdelta)

    return eps_zdelta, twiss_zdelta, enveloppes_zdelta


def beam_parameters_all(eps_zdelta, twiss_zdelta, gamma):
    """Convert the [z - delta] beam parameters in [phi - W] and [z - z']."""
    d_eps = _emittances_all(eps_zdelta, gamma)
    d_twiss = _twiss_all(twiss_zdelta, gamma)
    d_enveloppes = _enveloppes_all(d_twiss, d_eps)
    return d_eps, d_twiss, d_enveloppes


# =============================================================================
# Private - sigma
# =============================================================================
def _sigma_beam_matrices(arr_r_zz, sigma_in):
    """
    Compute the sigma beam matrices between linac entry and idx_out.

    sigma_in and transfer matrices should be in the same ref. By default,
    LW calculates transfer matrices in [z - delta].
    """
    l_sigma = []
    n_points = arr_r_zz.shape[0]

    for i in range(n_points):
        l_sigma.append(arr_r_zz[i] @ sigma_in @ arr_r_zz[i].transpose())
    return np.array(l_sigma)


# =============================================================================
# Private - emittance
# =============================================================================
def _emittance_zdelta(arr_sigma):
    """Compute longitudinal emittance, unnormalized, in pi.m.rad."""
    l_epsilon_zdelta = [np.sqrt(np.linalg.det(arr_sigma[i]))
                        for i in range(arr_sigma.shape[0])]
    return np.array(l_epsilon_zdelta)


def _emittances_all(eps_zdelta, gamma):
    """Compute emittances in [phi-W] and [z-z']."""
    d_eps = {"zdelta": eps_zdelta,
             "w": _convert_emittance(eps_zdelta, "zdelta to w", gamma),
             "z": _convert_emittance(eps_zdelta, "zdelta to z", gamma)}
    return d_eps


# TODO may be possible to save some operations by using lambda func?
def _convert_emittance(eps_orig, str_convert, gamma, beta=None,
                       lam=LAMBDA_BUNCH, e_0=E_rest_MeV):
    """Convert emittance from a phase space to another."""
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)

    # Lighten the dict
    gamma2 = gamma**2
    k_1 = 360. * e_0 * (gamma * beta) / lam
    k_2 = k_1 * gamma2

    # Dict of emittances conversion constants
    d_convert = {
        "w to z": 1. / k_2,
        "z to w": k_2,
        "w to zdelta": 1e-6 / k_1,
        "zdelta to w": 1e6 * k_1,
        "z to zdelta": 1e-6 * gamma2,
        "zdelta to z": 1e6 / gamma2,
    }
    eps_new = eps_orig * d_convert[str_convert]
    return eps_new


# =============================================================================
# Private - Twiss
# =============================================================================
def _twiss_zdelta(arr_sigma, arr_eps_zdelta):
    """Transport Twiss parameters along element(s) described by r_zz."""
    n_points = arr_sigma.shape[0]
    arr_twiss = np.full((n_points, 3), np.NaN)

    for i in range(n_points):
        sigma = arr_sigma[i]
        arr_twiss[i, :] = np.array([-sigma[1, 0],
                                    sigma[0, 0] * 10.,
                                    sigma[1, 1] / 10.]) / arr_eps_zdelta[i]
        # beta multiplied by 10 to match TW
        # gamma divided by 10 to keep beta * gamma - alpha**2 = 1
    return arr_twiss


def _twiss_all(twiss_zdelta, gamma):
    """Compute Twiss parameters in [phi-W] and [z-z']."""
    d_twiss = {"zdelta": twiss_zdelta,
               "w": _convert_twiss(twiss_zdelta, "zdelta to w", gamma),
               "z": _convert_twiss(twiss_zdelta, "zdelta to z", gamma)}
    return d_twiss


# TODO may be possible to save some operations by using lambda func?
def _convert_twiss(twiss_orig, str_convert, gamma, beta=None, lam=LAMBDA_BUNCH,
                   e_0=E_rest_MeV):
    """Convert Twiss array from a phase space to another."""
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)

    # Lighten the dict
    k_1 = e_0 * (gamma * beta) * lam / 360.
    k_2 = k_1 * beta**2
    k_3 = k_2 * gamma**2

    # Dict of emittances conversion constants
    d_convert = {
        "w to z": [-1., 1e-6 * k_3],
        "z to w": [-1., 1e6 / k_3],
        "w to zdelta": [-1., 1e-5 * k_2],
        "zdelta to w": [-1., 1e5 / k_2],
        "z to zdelta": [1., 1e1 * gamma**-2],
        "zdelta to z": [1., 1e-1 * gamma**2],
    }
    factors = d_convert[str_convert]

    # New array of Twiss parameters in the desired phase space
    twiss_new = np.empty(twiss_orig.shape)
    twiss_new[:, 0] = twiss_orig[:, 0] * factors[0]
    twiss_new[:, 1] = twiss_orig[:, 1] * factors[1]
    twiss_new[:, 2] = twiss_orig[:, 2] / factors[1]
    return twiss_new


# =============================================================================
# Private - Beam enveloppes
# =============================================================================
def _enveloppes(twiss, eps):
    """Compute beam enveloppes in a given plane."""
    env = np.sqrt(np.column_stack((twiss[:, 1], twiss[:, 2]) * eps))
    return env


def _enveloppes_all(twiss, eps):
    """Compute beam envelopes in all the planes."""
    d_env = {key: _enveloppes(twiss[key], eps[key]) for key in twiss.keys()}
    return d_env


# =============================================================================
# Private - to be moved to debug
# =============================================================================
# TODO Twiss should become Accelerator attribute, and plot_twiss should go in
# debug
def _plot_twiss(linac, twiss_zdelta):
    """Plot Twiss parameters."""
    _, axs = helper.create_fig_if_not_exist(33, [311, 312, 313])
    z_pos = linac.synch.z['abs_array']

    axs[0].plot(z_pos, twiss_zdelta[:, 0], label=linac.name)
    axs[0].set_ylabel(r'$\alpha_z$ [1]')
    axs[0].legend()

    axs[1].plot(z_pos, twiss_zdelta[:, 1])
    axs[1].set_ylabel(r'$\beta_z$ [mm/$\pi$%]')

    axs[2].plot(z_pos, twiss_zdelta[:, 2])
    axs[2].set_ylabel(r'$\gamma_z$ [$\pi$/mm/%]')
    axs[2].set_xlabel('s [m]')

    for ax_ in axs:
        ax_.grid(True)


# TODO Should also go in debug
def _output_twiss(d_twiss, idx=0):
    """Output Twiss parameters in three phase spaces at index idx."""
    d_units = {"zdelta": ["[1]", "[mm/pi.%]", "[pi/mm.%]"],
               "w": ["[1]", "[deg/pi.MeV]", "[pi/deg.MeV]"],
               "z": ["[1]", "[mm/pi.mrad]", "[pi/mm.mrad]"]}

    df_twiss = pd.DataFrame(columns=(
        'Twiss',
        '[z - delta]', 'Unit',
        '[phi - W]', 'Unit',
        "[z - z']", 'Unit'))

    for i, name in enumerate(["alpha", "beta", "gamma"]):
        line = [name]

        for phase_space in ["zdelta", "w", "z"]:
            line.append(d_twiss[phase_space][idx, i])
            line.append(d_units[phase_space][i])
        df_twiss.loc[i] = line

    df_twiss = df_twiss.round(decimals=4)
    pd.options.display.max_columns = 8
    pd.options.display.width = 120
    helper.printd(df_twiss, header=f"Twiss parameters at index {idx}:")


# =============================================================================
# Old junk
# =============================================================================
def plot_phase_spaces(linac, twiss):
    """Plot ellipsoid."""
    fig, ax = helper.create_fig_if_not_exist(34, [111])
    ax = ax[0]
    z_pos = linac.get_from_elements('pos_m', 'abs')

    ax.set_xlabel('z [m]')
    ax.set_ylabel("z' [m]")
    ax.grid(True)


def _beam_unnormalized_beam_emittance(w, w_prime):
    """Compute the beam rms unnormalized emittance (pi.m.rad)."""
    emitt_w = _beam_rms_size(w)**2 * _beam_rms_size(w_prime)**2
    emitt_w -= _compute_mean(w * w_prime)**2
    emitt_w = np.sqrt(emitt_w)
    return emitt_w


def beam_unnormalized_effective_emittance(w, w_prime):
    """Compute the beam rms effective emittance (?)."""
    return 5. * _beam_unnormalized_beam_emittance(w, w_prime)


def twiss_parameters(w, w_prime):
    """Compute the Twiss parameters."""
    emitt_w = _beam_unnormalized_beam_emittance(w, w_prime)
    alpha_w = -_beam_rms_correlation(w, w_prime) / emitt_w
    beta_w = _beam_rms_size(w)**2 / emitt_w
    gamma_w = _beam_rms_size(w_prime)**2 / emitt_w
    return alpha_w, beta_w, gamma_w


def _compute_mean(w):
    """Compute the mean value of a property over the beam at location s <w>."""
    return np.NaN


def _beam_rms_size(w):
    """Compute the beam rms size ~w."""
    return np.sqrt(_compute_mean((w - _compute_mean(w))**2))


def _beam_rms_correlation(w, v):
    """Compute the beam rms correlation bar(wv)."""
    return _compute_mean((w - _compute_mean(w)) * (v - _compute_mean(v)))


def calc_emittance_from_tw_transf_mat(linac, sigma_in):
    """Compute emittance with TW's transfer matrices."""
    l_sigma = [sigma_in]
    fold = linac.files['project_folder']
    filepath_ref = [fold + '/results/M_55_ref.txt',
                    fold + '/results/M_56_ref.txt',
                    fold + '/results/M_65_ref.txt',
                    fold + '/results/M_66_ref.txt']
    r_zz_tmp = tw.load_transfer_matrices(filepath_ref)
    z = r_zz_tmp[:, 0]
    n_z = z.shape[0]
    r_zz_ref = np.empty([n_z, 2, 2])
    for i in range(n_z):
        r_zz_ref[i, 0, 0] = r_zz_tmp[i, 1]
        r_zz_ref[i, 0, 1] = r_zz_tmp[i, 2]
        r_zz_ref[i, 1, 0] = r_zz_tmp[i, 3]
        r_zz_ref[i, 1, 1] = r_zz_tmp[i, 4]
        l_sigma.append(r_zz_ref[i] @ sigma_in @ r_zz_ref[i].transpose())
    arr_sigma = np.array(l_sigma)

    l_epsilon_zdelta = [np.sqrt(np.linalg.det(arr_sigma[i]))
                        for i in range(n_z)]
    arr_eps_zdelta = np.array(l_epsilon_zdelta)

    filepath = linac.files['project_folder'] + '/results/Chart_Energy(MeV).txt'
    w_kin = np.loadtxt(filepath, skiprows=1)
    w_kin = np.interp(x=z, xp=w_kin[:, 0], fp=w_kin[:, 1])
    gamma = helper.kin_to_gamma(w_kin)
    arr_eps_w = _convert_emittance(arr_eps_zdelta, "zdelta to w", gamma)

    fig, ax = helper.create_fig_if_not_exist(13, [111])
    ax = ax[0]
    ax.plot(z, arr_eps_w, label="Calc with TW transf mat")
    ax.legend()
