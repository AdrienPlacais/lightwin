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
from constants import E_REST_MEV, LAMBDA_BUNCH, SIGMA_ZDELTA


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
    envelopes_zdelta = _envelopes(twiss_zdelta, eps_zdelta)
    d_zdelta = {'twiss': twiss_zdelta,
                'eps': eps_zdelta,
                'envelopes': envelopes_zdelta}

    return d_zdelta


def beam_parameters_all(d_zdelta, gamma):
    """Convert the [z - delta] beam parameters in [phi - W] and [z - z']."""
    d_eps = _emittances_all(d_zdelta["eps"], gamma)
    d_twiss = _twiss_all(d_zdelta["twiss"], gamma)
    d_envelopes = _envelopes_all(d_twiss, d_eps)
    d_beam_parameters = {"twiss": d_twiss,
                         "eps": d_eps,
                         "envelopes": d_envelopes}
    return d_beam_parameters


def mismatch_factor(twiss_ref, twiss_fix, transp=False):
    """Compute the mismatch factor between two ellipses."""
    # Transposition needed when computing the mismatch factor at more than one
    # position, as shape of twiss arrays is (n, 3).
    if transp:
        twiss_ref = twiss_ref.transpose()
        twiss_fix = twiss_fix.transpose()

    # R in TW doc
    __r = twiss_ref[1] * twiss_fix[2] + twiss_ref[2] * twiss_fix[1]
    __r -= 2. * twiss_ref[0] * twiss_fix[0]

    # Forbid R values lower than 2 (numerical error)
    __r = np.atleast_1d(__r)
    __r[np.where(__r < 2.)] = 2.

    mismatch = np.sqrt(.5 * (__r + np.sqrt(__r**2 - 4.))) - 1.
    return mismatch

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
    d_eps = {"eps_zdelta": eps_zdelta,
             "eps_w": _convert_emittance(eps_zdelta, "zdelta to w", gamma),
             "eps_z": _convert_emittance(eps_zdelta, "zdelta to z", gamma)}
    return d_eps


# TODO may be possible to save some operations by using lambda func?
def _convert_emittance(eps_orig, str_convert, gamma, beta=None,
                       lam=LAMBDA_BUNCH, e_0=E_REST_MEV):
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
    d_twiss = {"twiss_zdelta": twiss_zdelta,
               "twiss_w": _convert_twiss(twiss_zdelta, "zdelta to w", gamma),
               "twiss_z": _convert_twiss(twiss_zdelta, "zdelta to z", gamma)}
    return d_twiss


# TODO may be possible to save some operations by using lambda func?
def _convert_twiss(twiss_orig, str_convert, gamma, beta=None, lam=LAMBDA_BUNCH,
                   e_0=E_REST_MEV):
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
# Private - Beam envelopes
# =============================================================================
def _envelopes(twiss, eps):
    """Compute beam envelopes in a given plane."""
    env = np.sqrt(np.column_stack((twiss[:, 1], twiss[:, 2]) * eps))
    return env


def _envelopes_all(twiss, eps):
    """Compute beam envelopes in all the planes."""
    spa = ['_zdelta', '_w', '_z']
    d_env = {'envelopes' + key:
             _envelopes(twiss['twiss' + key], eps['eps' + key])
             for key in spa}
    return d_env

