#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 14:42:41 2021.

@author: placais

This module holds everything related to emittances, Twiss parameters,
envelopes.

Conventions
-----------
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
"""
from typing import Any
from dataclasses import dataclass

import numpy as np

import config_manager as con
from core.elements import _Element
import util.converters as convert
from util.helper import recursive_items, recursive_getter, range_vals


@dataclass
class BeamParameters:
    """Hold all emittances, envelopes, etc in various planes."""

    tm_cumul: np.ndarray
    sigma_in: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Create emittance and twiss in the [z-delta] plane."""
        if self.sigma_in is None:
            self.sigma_in = con.SIGMA_ZDELTA

        self.sigma = _sigma_beam_matrices(self.tm_cumul, self.sigma_in)
        self.eps_zdelta = _emittance_zdelta(self.sigma)
        self.twiss_zdelta = _twiss_zdelta(self.sigma, self.eps_zdelta)

        self.mismatch_factor: np.ndarray | None

        self.alpha_zdelta: np.ndarray
        self.beta_zdelta: np.ndarray
        self.gamma_zdelta: np.ndarray
        self.alpha_z: np.ndarray
        self.beta_z: np.ndarray
        self.gamma_z: np.ndarray
        self.alpha_w: np.ndarray
        self.beta_w: np.ndarray
        self.gamma_w: np.ndarray

        self.envelope_pos_zdelta: np.ndarray
        self.envelope_energy_zdelta: np.ndarray
        self.envelope_pos_z: np.ndarray
        self.envelope_energy_z: np.ndarray
        self.envelope_pos_w: np.ndarray
        self.envelope_energy_w: np.ndarray

    def __str__(self) -> str:
        """Give compact information on the data that is stored."""
        out = "\tBeamParameters:\n"
        out += "\t\t" + range_vals("eps_zdelta", self.eps_zdelta)
        out += "\t\t" + range_vals("beta_zdelta", self.beta_zdelta)
        out += "\t\t" + range_vals("mismatch", self.mismatch_factor)
        return out

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True,
            elt: _Element | None = None, pos: str | None = None,
            **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        elt : _Element | None, optional
            If provided, return the attributes only at the considered _Element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            _Element.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = recursive_getter(key, vars(self), **kwargs)
            if not to_numpy and isinstance(val[key], np.ndarray):
                val[key] = val[key].tolist()

            # if None not in (self.element_to_index, elt):
                # idx = self.element_to_index(elt, pos)
                # val[key] = val[key][idx]

        out = [np.array(val[key]) if to_numpy and not isinstance(val[key], str)
               else val[key]
               for key in keys]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def compute_mismatch(self, ref_twiss_zdelta: np.ndarray | None) -> None:
        """Compute the mismatch factor."""
        self.mismatch_factor = None
        if ref_twiss_zdelta is not None:
            self.mismatch_factor = mismatch_factor(ref_twiss_zdelta,
                                                   self.twiss_zdelta,
                                                   transp=True)

    def compute_full(self, gamma: np.ndarray) -> None:
        """Compute emittances, Twiss, envelopes in every plane."""
        _eps = _emittances_all(self.eps_zdelta, gamma)
        self.eps_w = _eps['eps_w']
        self.eps_z = _eps['eps_z']

        _twiss = _twiss_all(self.twiss_zdelta, gamma)
        self.alpha_zdelta = _twiss['twiss_zdelta'][:, 0]
        self.beta_zdelta = _twiss['twiss_zdelta'][:, 1]
        self.gamma_zdelta = _twiss['twiss_zdelta'][:, 2]
        self.alpha_z = _twiss['twiss_z'][:, 0]
        self.beta_z = _twiss['twiss_z'][:, 1]
        self.gamma_z = _twiss['twiss_z'][:, 2]
        self.alpha_w = _twiss['twiss_w'][:, 0]
        self.beta_w = _twiss['twiss_w'][:, 1]
        self.gamma_w = _twiss['twiss_w'][:, 2]

        _envelopes = _envelopes_all(_twiss, _eps)
        self.envelope_pos_zdelta = _envelopes['envelopes_zdelta'][:, 0]
        self.envelope_energy_zdelta = _envelopes['envelopes_zdelta'][:, 1]
        self.envelope_pos_z = _envelopes['envelopes_z'][:, 0]
        self.envelope_energy_z = _envelopes['envelopes_z'][:, 1]
        self.envelope_pos_w = _envelopes['envelopes_w'][:, 0]
        self.envelope_energy_w = _envelopes['envelopes_w'][:, 1]


# =============================================================================
# Public
# =============================================================================
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
def _sigma_beam_matrices(tm_cumul: np.ndarray, sigma_in: np.ndarray
                         ) -> np.ndarray:
    """
    Compute the sigma beam matrices between over the linac.

    sigma_in and transfer matrices should be in the same ref. By default,
    LW calculates transfer matrices in [z - delta].
    """
    sigma = []
    n_points = tm_cumul.shape[0]

    for i in range(n_points):
        sigma.append(tm_cumul[i] @ sigma_in @ tm_cumul[i].transpose())
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
    """Transport Twiss parameters along element(s) described by tm_cumul."""
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
