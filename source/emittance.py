#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:42:41 2021.

@author: placais

TODO : check transport of [sigma] and Twiss. Not sure about initial vector

Longitudinal emittances:
    eps_z is emittance in [z-z'] in [pi.mm.mrad], non normalized
    eps_zdelta is emittance in [z-delta] in [pi.m.rad], non normalized
    eps_w is emittance in [Delta phi-Delta W] in [pi.deg.MeV], normalized


Twiss:
    beta, gamma are Lorentz factors.
    beta_z is beta Twiss parameter in [z-z'] [mm/pi.mrad]
    beta_zdelta is beta Twiss parameter in [z-delta] [mm/pi.%]
    beta_w is beta Twiss parameter in [Delta phi-Delta W] [deg/pi.MeV]
    (same for gamma_z, gamma_z, gamma_zdelta)

    Conversions for alpha are easier:
        alpha_w = -alpha_z = -alpha_zdelta
"""

import numpy as np
import helper
import accelerator
from constants import E_rest_MeV, LAMBDA_BUNCH
import tracewin_interface as tw


def sigma_beam_matrices2(arr_r_zz, sigma_in):
    """
    Compute the sigma beam matrices corresponding to provided transfer matrix.

    Parameters
    ----------
    arr_r_zz : numpy array
        (n, 2, 2) array of total longitudinal transfer matrices.
    sigma_in : numpy array
        (2, 2) sigma beam matrix at the entry of the linac.

    Return
    ------
    arr_sigma : numpy array
        (n, 2, 2) sigma beam matrices.
    """
    n_points = arr_r_zz.shape[0]
    arr_sigma = np.empty((n_points))
    for i in range(n_points):
        arr_sigma[i] = arr_r_zz[i] @ sigma_in @ arr_r_zz[i].transpose()
    return arr_sigma


def sigma_beam_matrices(linac, sigma_in, idx_out=-1):
    """
    Compute the sigma beam matrices between linac entry and idx_out.

    sigma_in and transfer matrices should be in the same ref. By default,
    LW calculates transfer matrices in [z - delta].
    """
    l_sigma = [sigma_in]
    n_points = linac.transf_mat['cumul'][0:idx_out].shape[0] + 1

    for i in range(n_points):
        r_zz = linac.transf_mat['cumul'][i]
        l_sigma.append(r_zz @ sigma_in @ r_zz.transpose())
    return np.array(l_sigma)


def emittance_zdelta(linac, sigma_in, idx_out=-1):
    """Compute longitudinal emittance, unnormalized, in pi.m.rad."""
    arr_sigma = sigma_beam_matrices(linac, sigma_in, idx_out)

    l_epsilon_zdelta = [np.sqrt(np.linalg.det(arr_sigma[i]))
                        for i in range(arr_sigma.shape[0])]
    return np.array(l_epsilon_zdelta)


def plot_longitudinal_emittance(linac, sigma_in):
    """Compute and plot longitudinal emittance."""
    # Array of non normalized emittance in [z-delta]
    gamma = linac.synch.energy['gamma_array']

    arr_eps_zdelta = emittance_zdelta(linac, sigma_in, idx_out=-1)[1:]

    arr_eps_w = eps_zdelta_to_w(arr_eps_zdelta, gamma)

    fig, axx = helper.create_fig_if_not_exist(13, [111])
    axx = axx[0]
    axx.plot(linac.synch.z['abs_array'], arr_eps_w, label=linac.name)
    axx.grid(True)
    axx.set_xlabel("Position [m]")
    axx.set_ylabel(r"Longitudinal emittance [$\pi$.deg.MeV]")
    axx.legend()
    print("plot_longitudinal_emittance: normalized or not?",
          "Should increase slowly.")


def _transform_mt(transfer_matrix):
    """Change form of the transfer matrix."""
    n_points = transfer_matrix.shape[0]
    transformed = np.full((n_points, 3, 3), np.NaN)
    for i in range(n_points):
        C = transfer_matrix[i, 0, 0]
        C_prime = transfer_matrix[i, 1, 0]
        S = transfer_matrix[i, 0, 1]
        S_prime = transfer_matrix[i, 1, 1]

        transformed[i, :, :] = np.array((
            [C**2, -2. * C * S, S**2],
            [-C * C_prime, C_prime * S + C * S_prime, -S * S_prime],
            [C_prime**2, -2. * C_prime * S_prime, S_prime**2]))
    return transformed


def transport_twiss_parameters2(r_zz, alpha_z0, beta_z0):
    """Transport Twiss parameters along element(s) described by r_zz."""
    n_points = r_zz.shape[0]
    transformed = _transform_mt(r_zz)
    arr_twiss = np.full((n_points, 3), np.NaN)
    arr_twiss[0, :] = np.array(([beta_z0,
                                 alpha_z0,
                                 (1. + alpha_z0**2) / beta_z0]))

    for i in range(1, n_points):
        arr_twiss[i, :] = transformed[i, :, :] @ arr_twiss[0, :]
    return arr_twiss


def transport_twiss_parameters(linac, alpha_z0, beta_z0):
    """Transport Twiss parameters."""
    assert isinstance(linac, accelerator.Accelerator)
    t_m = linac.transf_mat['cumul']
    n_points = t_m.shape[0]

    transformed = _transform_mt(t_m, n_points)

    twiss = np.full((n_points, 3), np.NaN)
    twiss[0, :] = np.array(([beta_z0, alpha_z0, (1. + alpha_z0**2) / beta_z0]))

    for i in range(1, n_points):
        twiss[i, :] = transformed[i, :, :] @ twiss[0, :]
    return twiss


# =============================================================================
# Emittances conversions
# =============================================================================
def eps_z_to_w(eps_z, gamma, beta=None, lam=LAMBDA_BUNCH, e_0=E_rest_MeV):
    """Convert emittance from [z-z'] to [Delta phi-Delta W]."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)
    eps_w = (360. * e_0 * beta * gamma**3) / lam * eps_z
    return eps_w


def eps_w_to_z(eps_w, gamma, beta=None, lam=LAMBDA_BUNCH, e_0=E_rest_MeV):
    """Convert emittance from [Delta phi-Delta W] to [z-z']."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)
    eps_z = lam / (360. * e_0 * beta * gamma**3) * eps_w
    return eps_z


def eps_zdelta_to_w(eps_zdelta, gamma, beta=None, lam=LAMBDA_BUNCH,
                    e_0=E_rest_MeV):
    """Convert emittance from [z-delta] to [Delta phi-Delta W]. Validated."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)
    eps_w = (360. * e_0 * beta * gamma) / lam * eps_zdelta * 1e6
    return eps_w


def eps_w_to_zdelta(eps_w, gamma, beta=None, lam=LAMBDA_BUNCH,
                    e_0=E_rest_MeV):
    """Convert emittance from [Delta phi-Delta W] to [z-delta]."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)
    eps_zdelta = lam / (360. * e_0 * beta * gamma) * eps_w * 1e-6
    return eps_zdelta


# =============================================================================
# Twiss beta conversions
# =============================================================================
def beta_z_to_w(beta_z, gamma, beta=None, lam=LAMBDA_BUNCH,
                e_0=E_rest_MeV):
    """Convert Twiss beta from [z-z'] to [Delta phi-Delta W]. Validated."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)

    beta_w = 360. / (e_0 * (gamma * beta)**3 * lam) * beta_z * 1e6
    return beta_w


def beta_w_to_z(beta_w, gamma, beta=None, lam=LAMBDA_BUNCH,
                e_0=E_rest_MeV):
    """Convert Twiss beta from [Delta phi-Delta W] to [z-z']. Validated."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)
    beta_z = (e_0 * (gamma * beta)**3 * lam) / 360. * beta_w * 1e-6
    return beta_z


def beta_zdelta_to_w(beta_zdelta, gamma, beta=None, lam=LAMBDA_BUNCH,
                     e_0=E_rest_MeV):
    """Convert Twiss beta from [z-delta] to [Delta phi-Delta W]. Validated."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)

    beta_w = 360. / (e_0 * gamma * beta**3 * lam) * beta_zdelta * 1e5
    return beta_w


def beta_w_to_zdelta(beta_w, gamma, beta=None, lam=LAMBDA_BUNCH,
                     e_0=E_rest_MeV):
    """Convert Twiss beta [Delta phi-Delta W] from to [z-delta]. Validated."""
    # beta is Lorentz factor
    if beta is None:
        beta = np.sqrt(1. - gamma**-2)

    beta_zdelta = (e_0 * gamma * beta**3 * lam) / 360. * beta_w * 1e-5
    return beta_zdelta


def beta_z_to_zdelta(beta_z, gamma):
    """Convert Twiss beta [z-z'] from to [z-delta]. Validated."""
    beta_zdelta = beta_z * gamma**-2 * 1e1
    return beta_zdelta


def beta_zdelta_to_z(beta_zdelta, gamma):
    """Convert Twiss beta [z-delta] from to [z-z']. Validated."""
    beta_z = beta_zdelta * gamma**2 * 1e-1
    return beta_z


# =============================================================================
# Twiss gamma functions (inverse functions of Twiss beta functions!)
# =============================================================================
def gamma_z_to_w(gamma_z, gamma, beta=None, lam=LAMBDA_BUNCH,
                 e_0=E_rest_MeV):
    """Convert Twiss gamma from [z-z'] to [Delta phi-Delta W]."""
    gamma_w = beta_w_to_z(gamma_z, gamma, beta, lam, e_0)
    return gamma_w


def gamma_w_to_z(gamma_w, gamma, beta=None, lam=LAMBDA_BUNCH,
                 e_0=E_rest_MeV):
    """Convert Twiss gamma from [Delta phi-Delta W] to [z-z']."""
    gamma_z = beta_z_to_w(gamma_w, gamma, beta, lam, e_0)
    return gamma_z


def gamma_zdelta_to_w(gamma_zdelta, gamma, beta=None, lam=LAMBDA_BUNCH,
                      e_0=E_rest_MeV):
    """Convert Twiss gamma from [z-delta] to [Delta phi-Delta W]."""
    gamma_w = beta_w_to_zdelta(gamma_zdelta, gamma, beta, lam, e_0)
    return gamma_w


def gamma_w_to_zdelta(gamma_w, gamma, beta=None, lam=LAMBDA_BUNCH,
                      e_0=E_rest_MeV):
    """Convert Twiss gamma [Delta phi-Delta W] from to [z-delta]."""
    gamma_zdelta = beta_zdelta_to_w(gamma_w, gamma, beta, lam, e_0)
    return gamma_zdelta


def gamma_z_to_zdelta(gamma_z, gamma):
    """Convert Twiss gamma [z-z'] from to [z-delta]."""
    gamma_zdelta = gamma_z * gamma**-2 * 1e1
    return gamma_zdelta


def gamma_zdelta_to_z(gamma_zdelta, gamma):
    """Convert Twiss beta [z-delta] from to [z-z']."""
    gamma_z = gamma_zdelta * gamma**2 * 1e-1
    return gamma_z


# =============================================================================
# Old junk
# =============================================================================
def plot_twiss(linac, twiss):
    """Plot Twiss parameters."""
    fig, ax = helper.create_fig_if_not_exist(33, [111])
    ax = ax[0]
    z_pos = linac.get_from_elements('pos_m', 'abs')
    ax.plot(z_pos, twiss[:, 1], label=r'$\alpha_z$')
    ax.plot(z_pos, twiss[:, 0], label=r'$\beta_z$')
    ax.plot(z_pos, twiss[:, 2], label=r'$\gamma_z$')
    ax.set_xlabel('s [m]')
    ax.set_ylabel('Twiss parameters')
    ax.grid(True)
    ax.legend()


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
    arr_eps_w = eps_zdelta_to_w(arr_eps_zdelta, gamma)

    fig, ax = helper.create_fig_if_not_exist(13, [111])
    ax = ax[0]
    ax.plot(z, arr_eps_w, label="Calc with TW transf mat")
    ax.legend()
