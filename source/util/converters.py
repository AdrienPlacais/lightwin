#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""All functions to change units."""
import numpy as np

from constants import c
import config_manager as con


def position(pos_in: float | np.ndarray, beta: float | np.ndarray, key: str,
             omega: float | None = None) -> float | np.ndarray:
    """Phase/position converters."""
    if omega is None:
        omega = con.OMEGA_0_BUNCH
    conversion_functions = {
        "z to phi": lambda pos, bet: -omega * pos / (bet * c),
        "phi to z": lambda pos, bet: -pos * bet * c / omega,
    }
    return conversion_functions[key](pos_in, beta)


def energy(energy_in: float | np.ndarray, key: str,
           q_over_m: float | None = None, m_over_q: float | None = None,
           e_rest: float | None = None) -> float | np.ndarray:
    """Convert energy or Lorentz factor into another related quantity."""
    if q_over_m is None:
        q_over_m = con.Q_OVER_M
    if m_over_q is None:
        m_over_q = con.M_OVER_Q
    if e_rest is None:
        e_rest = con.E_REST_MEV

    conversion_functions = {
        "v to kin": lambda x: 0.5 * m_over_q * x**2 * 1e-6,
        "kin to v": lambda x: np.sqrt(2e6 * q_over_m * x),
        "kin to gamma": lambda x: 1. + x / e_rest,
        "gamma to kin": lambda x: e_rest * (x - 1.),
        "beta to gamma": lambda x: 1. / np.sqrt(1. - x**2),
        "gamma to beta": lambda x: np.sqrt(1. - x**-2),
        "kin to beta": lambda x: np.sqrt(1. - (e_rest / (x + e_rest))**2),
        "beta to kin": lambda x: None,
        "kin to p": lambda x: np.sqrt((x + e_rest)**2 - e_rest**2),
        "p to kin": lambda x: np.sqrt(x**2 + e_rest**2) - e_rest,
        "gamma to p": lambda x: x * np.sqrt(1. - x**-2) * e_rest,
        "beta to p": lambda x: x / np.sqrt(1. - x**2) * e_rest,
    }
    return conversion_functions[key](energy_in)


def longitudinal(long_in: float | np.ndarray, ene: float | np.ndarray,
                 key: str, e_rest: float | None = None) -> float | np.ndarray:
    """Convert energies between longitudinal phase spaces."""
    if e_rest is None:
        e_rest = con.E_REST_MEV
    conversion_functions = {
        "zprime gamma to zdelta": lambda zp, gam: zp * gam**-2 * 1e-1,
        "zprime kin to zdelta": lambda zp, kin:
            zp * (1. + kin / e_rest)**-2 * 1e-1,
    }
    return conversion_functions[key](long_in, ene)


# TODO may be possible to save some operations by using lambda func?
def emittance(eps_orig: float | np.ndarray, key: str,
              gamma_kin: float | np.ndarray | None = None,
              beta_kin: float | np.ndarray | None = None,
              lam: float | np.ndarray | None = None,
              e_0: float | np.ndarray | None = None) -> float | np.ndarray:
    """Convert emittance from a phase space to another, or handle norm."""
    if lam is None:
        lam = con.LAMBDA_BUNCH
    if e_0 is None:
        e_0 = con.E_REST_MEV
    k_1 = 360. * e_0 / lam
    k_2 = gamma_kin * beta_kin
    k_3 = k_2 * gamma_kin**2

    conversion_constants = {
        "phiw to z": 1. / k_1,
        "z to phiw": k_1,
        "phiw to zdelta": 0.1 / k_1,
        "zdelta to phiw": 10 * k_1,
        "z to zdelta": 0.1,
        "zdelta to z": 10.,
        "normalize zdelta": k_2,
        "de-normalize zdelta": 1. / k_2,
        "normalize phiw": k_2,
        "de-normalize phiw": 1. / k_2,
        "normalize z": k_3,
        "de-normalize z": 1. / k_3,
        "normalize x": k_2,
        "de-normalize x": 1. / k_2,
        "normalize y": k_2,
        "de-normalize y": 1. / k_2,
        "normalize x99": k_2,
        "de-normalize x99": 1. / k_2,
        "normalize y99": k_2,
        "de-normalize y99": 1. / k_2,
    }
    eps_new = eps_orig * conversion_constants[key]
    return eps_new


def twiss(twiss_orig: np.ndarray, gamma_kin: float | np.ndarray, key: str,
          lam: float | np.ndarray | None = None,
          e_0: float | np.ndarray | None = None,
          beta_kin: float | np.ndarray | None = None) -> np.ndarray:
    """Convert Twiss array from a phase space to another."""
    if lam is None:
        lam = con.LAMBDA_BUNCH
    if e_0 is None:
        e_0 = con.E_REST_MEV
    if beta_kin is None:
        beta_kin = np.sqrt(1. - gamma_kin**-2)

    # Lighten the dict
    k_1 = e_0 * (gamma_kin * beta_kin) * lam / 360.
    k_2 = k_1 * beta_kin**2
    k_3 = k_2 * gamma_kin**2

    conversion_constants = {
        "phiw to z": [-1., 1e-6 * k_3],
        "z to phiw": [-1., 1e6 / k_3],
        "phiw to zdelta": [-1., 1e-5 * k_2],
        "zdelta to phiw": [-1., 1e5 / k_2],
        "z to zdelta": [1., 1e1 * gamma_kin**-2],
        "zdelta to z": [1., 1e-1 * gamma_kin**2],
    }
    factors = conversion_constants[key]

    # New array of Twiss parameters in the desired phase space
    twiss_new = np.empty(twiss_orig.shape)
    twiss_new[:, 0] = twiss_orig[:, 0] * factors[0]
    twiss_new[:, 1] = twiss_orig[:, 1] * factors[1]
    twiss_new[:, 2] = twiss_orig[:, 2] / factors[1]
    return twiss_new
