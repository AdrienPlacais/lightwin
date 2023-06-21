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

import config_manager as con
import util.converters as convert


# =============================================================================
# Public
# =============================================================================
def beam_parameters_zdelta(r_zz: np.ndarray, sigma_in: np.ndarray | None = None
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sigma beam matrix, emittance, Twiss parameters.

    Parameters
    ----------
    r_zz : np.ndarray
        (n, 2, 2) cumulated transfer matrices.
    sigma_in : np.ndarray | None, optional
        (2, 2) sigma beam matrix at entry of linac. The default is None. In
        this case, we take the sigma matrix provided by the user in the
        configuration file.

    Returns
    -------
    eps_zdelta : np.ndarray
        (n) array of emittances.
    twiss_zdelta : np.ndarray
        (n, 3) Twiss parameters (alfa, beta, gamma).
    sigma : np.ndarray
        (n, 2, 2) sigma beam matrices.
    """
    if sigma_in is None:
        sigma_in = con.SIGMA_ZDELTA
    sigma = _sigma_beam_matrices(r_zz, sigma_in)

    eps_zdelta = _emittance_zdelta(sigma)
    twiss_zdelta = _twiss_zdelta(sigma, eps_zdelta)
    return eps_zdelta, twiss_zdelta, sigma


def beam_parameters_all(eps_zdelta: np.ndarray, twiss_zdelta: np.ndarray,
                        gamma: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """
    Convert the [z - delta] beam parameters in [phi - W] and [z - z'].

    Parameters
    ----------
    eps_zdelta : np.ndarray
        Longitudinal emittances in the z-delta plane.
    twiss_zdelta : np.ndarray
        Twiss parameters (alfa, beta, gamma) in the z-delta plane.
    gamma : np.ndarray
        Lorentz gamma factor.

    Returns
    -------
    beam_parameters : dict[str, dict[str, np.ndarray]]
        A dict holding the Twiss parameters, the emittances, and the envelopes.
        Each key is a dictionary holding these quantities in the different
        longitudinal phase spaces.

    """
    eps = _emittances_all(eps_zdelta, gamma)
    twiss = _twiss_all(twiss_zdelta, gamma)
    envelopes = _envelopes_all(twiss, eps)
    beam_parameters = {"twiss": twiss,
                       "eps": eps,
                       "envelopes": envelopes}
    return beam_parameters


def mismatch_factor(ref: np.ndarray, fix: np.ndarray, transp: bool = False
                    ) -> float:
    """Compute the mismatch factor between two ellipses."""
    assert isinstance(ref, np.ndarray)
    assert isinstance(fix, np.ndarray)
    # Transposition needed when computing the mismatch factor at more than one
    # position, as shape of twiss arrays is (n, 3).
    if transp:
        ref = ref.transpose()
        fix = fix.transpose()

    # R in TW doc
    __r = ref[1] * fix[2] + ref[2] * fix[1]
    __r -= 2. * ref[0] * fix[0]

    # Forbid R values lower than 2 (numerical error)
    __r = np.atleast_1d(__r)
    __r[np.where(__r < 2.)] = 2.

    mismatch = np.sqrt(.5 * (__r + np.sqrt(__r**2 - 4.))) - 1.
    return mismatch


# =============================================================================
# Private
# =============================================================================
def _sigma_beam_matrices(r_zz: np.ndarray, sigma_in: np.ndarray) -> np.ndarray:
    """
    Compute the sigma beam matrices between linac entry and idx_out.

    sigma_in and transfer matrices should be in the same ref. By default,
    LW calculates transfer matrices in [z - delta].
    """
    sigma = []
    n_points = r_zz.shape[0]

    for i in range(n_points):
        sigma.append(r_zz[i] @ sigma_in @ r_zz[i].transpose())
    return np.array(sigma)


def _emittance_zdelta(sigma: np.ndarray) -> np.ndarray:
    """Compute longitudinal emittance, unnormalized, in pi.m.rad."""
    epsilon_zdelta = [np.sqrt(np.linalg.det(sigma[i]))
                      for i in range(sigma.shape[0])]
    return np.array(epsilon_zdelta)


def _emittances_all(eps_zdelta: np.ndarray, gamma: np.ndarray
                    ) -> dict[str, np.ndarray]:
    """Compute emittances in [phi-W] and [z-z']."""
    eps = {"eps_zdelta": eps_zdelta,
           "eps_w": convert.emittance(eps_zdelta, gamma, "zdelta to w"),
           "eps_z": convert.emittance(eps_zdelta, gamma, "zdelta to z")}
    return eps


def _twiss_zdelta(sigma: np.ndarray, eps_zdelta: np.ndarray) -> np.ndarray:
    """Transport Twiss parameters along element(s) described by r_zz."""
    n_points = sigma.shape[0]
    twiss = np.full((n_points, 3), np.NaN)

    for i in range(n_points):
        twiss[i, :] = np.array([-sigma[i][1, 0],
                                sigma[i][0, 0] * 10.,
                                sigma[i][1, 1] / 10.]) / eps_zdelta[i]
        # beta multiplied by 10 to match TW
        # gamma divided by 10 to keep beta * gamma - alpha**2 = 1
    return twiss


def _twiss_all(twiss_zdelta: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Compute Twiss parameters in [phi-W] and [z-z']."""
    twiss = {"twiss_zdelta": twiss_zdelta,
             "twiss_w": convert.twiss(twiss_zdelta, gamma, "zdelta to w"),
             "twiss_z": convert.twiss(twiss_zdelta, gamma, "zdelta to z")}
    return twiss


def _envelopes(twiss: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Compute beam envelopes in a given plane."""
    env = np.sqrt(np.column_stack((twiss[:, 1], twiss[:, 2]) * eps))
    return env


def _envelopes_all(twiss: np.ndarray, eps: np.ndarray) -> np.ndarray:
    """Compute beam envelopes in all the planes."""
    spa = ['_zdelta', '_w', '_z']
    env = {'envelopes' + key:
           _envelopes(twiss['twiss' + key], eps['eps' + key])
           for key in spa}
    return env
