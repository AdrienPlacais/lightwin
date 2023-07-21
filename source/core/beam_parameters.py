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
from typing import Any, Callable
from dataclasses import dataclass
import logging

import numpy as np

import config_manager as con
from core.elements import _Element
import util.converters as converters
from util.helper import recursive_items, recursive_getter, range_vals


PHASE_SPACES = ['zdelta', 'z', 'phiw', 'phiw99',
                'x', 'y', 'x99', 'y99']


# FIXME avoid initialisation with sigma etc when from TraceWin
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

        self.zdelta = SinglePhaseSpaceBeamParameters(phase_space='zdelta')
        self.z = SinglePhaseSpaceBeamParameters(phase_space='z')
        self.phiw = SinglePhaseSpaceBeamParameters(phase_space='phiw')
        self.x = SinglePhaseSpaceBeamParameters(phase_space='x')
        self.y = SinglePhaseSpaceBeamParameters(phase_space='y')

        self.zdelta.init_from_sigma(self.sigma)
        self.mismatch_factor: np.ndarray | None = None

    def __str__(self) -> str:
        """Give compact information on the data that is stored."""
        out = "\tBeamParameters:\n"
        out += "\t\t" + range_vals("zdelta.eps", self.zdelta.eps)
        out += "\t\t" + range_vals("zdelta.beta", self.zdelta.beta)
        out += "\t\t" + range_vals("mismatch", self.mismatch_factor)
        return out

    def has(self, key: str) -> bool:
        """
        Tell if the required attribute is in this class.

        Specifics of this method: twiss_zdelta will return True, even if the
        correct property is zdelta.twiss.
        """
        if _phase_space_name_hidden_in_key(key):
            key, phase_space = _separate_var_from_phase_space(key)
            return key in recursive_items(vars(vars(self)[phase_space]))
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: _Element | None = None, pos: str | None = None,
            phase_space: str | None = None, **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        What is particular in this getter is that all
        SinglePhaseSpaceBeamParameters attributes have attributes with the same
        name: `twiss`, `alpha`, `beta`, `gamma`, `eps`, `envelopes_pos` and
        `envelopes_energy`.
        Hence, you must provide either a `phase_space` argument which shall be
        in PHASE_SPACES, either you must append the name of the phase space to
        the name of the desired variable.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        none_to_nan : bool, optional
            To convert None to np.NaN. The default is True.
        elt : _Element | None, optional
            If provided, return the attributes only at the considered _Element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            _Element.
        phase_space : ['z', 'zdelta', 'phi_w', 'x', 'y'] | None, optional
            Phase space in which you want the key. The default is None. In this
            case, the quantities from the zdelta phase space are taken.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}
        for key in keys:
            short_key = key
            if _phase_space_name_hidden_in_key(key):
                if phase_space is not None:
                    logging.warning(
                        "Amibiguous: you asked for two phase-spaces. One with "
                        f"keyword argument {phase_space = }, and another with "
                        f"the positional argument {key = }. I take phase "
                        f"space from {key = }.")
                short_key, phase_space = _separate_var_from_phase_space(key)

            if not self.has(short_key):
                val[key] = None
                continue

            for stored_key, stored_val in vars(self).items():
                if stored_key == short_key:
                    val[key] = stored_val
                    break

                if stored_key == phase_space:
                    val[key] = recursive_getter(
                        short_key, vars(stored_val), to_numpy=False,
                        none_to_nan=False, elt=elt, pos=pos, **kwargs)

                    if val[key] is None:
                        continue
                    break

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def init_zdelta_from_sigma(self) -> None:
        """Call the proper method to initialize zdelta."""
        self.zdelta.init_from_sigma(self.sigma)

    def init_other_longitudinal_planes_from_zdelta(self, gamma_kin: np.ndarray
                                                   ) -> None:
        """Create the other longitudinal planes from zdelta."""
        args = (self.zdelta.eps, self.zdelta.twiss, gamma_kin)
        self.phiw.init_from_another_plane(*args, 'zdelta to phiw')
        self.z.init_from_another_plane(*args, 'zdelta to z')

    def init_other_longitudinal_planes_from_zdelta_no_twiss(
            self, gamma_kin: np.ndarray) -> None:
        """Create the other longitudinal planes from zdelta (for TraceWin)."""
        args = (self.zdelta.eps, self.zdelta.envelope_pos,
                self.zdelta.envelope_energy, gamma_kin)
        self.phiw.init_from_another_plane_no_twiss(*args, 'zdelta to phiw')
        self.z.init_from_another_plane_no_twiss(*args, 'zdelta to z')

# FIXME will not work as for now. Tmp replaced by phiW only
    def init_all_phase_spaces_from_a_dict(
        self, results: dict[str, np.ndarray],
        results_converter: dict[str, Callable[[str], tuple[np.ndarray]]]
    ) -> None:
        """
        Init phase spaces from a dict, such as loaded after TW simulation.

        Parameters
        ----------
        results : dict[str, np.ndarray]
            Dictionary holding beam parameters.
        results_converter : dict[str, Callable[[str], tuple[np.ndarray]]]
            Dictionary to convert `results` into beam parameters. Keys must be
            phase-spaces names, values functions taking `results` in argument
            and returning a tuple holding (alpha, beta, gamma, eps,
            envelope_pos, envelope_energy.

        """
        for phase_space, converter in results_converter.items():
            alpha, beta, gamma, eps, envelope_pos, envelope_energy = \
                converter(results)
            self.phase_space = \
                SinglePhaseSpaceBeamParameters(phase_space=phase_space)

    def init_zdelta_from_dict(self, results: dict[str, np.ndarray]) -> None:
        """
        Init phiw from a dict, such as loaded after TW simulation.

        Parameters
        ----------
        results : dict[str, np.ndarray]
            Dictionary holding beam parameters.

        """
        self.zdelta.eps = results['ezdp']
        self.zdelta.envelope_pos = results['SizeZ']
        self.zdelta.envelope_energy = results['szdp']

    def compute_mismatch(self, ref_twiss_zdelta: np.ndarray | None) -> None:
        """Compute the mismatch factor."""
        if ref_twiss_zdelta is None:
            logging.warning("Attempting to compute a mismatch without "
                            "reference Twiss parameters.")
            return
        self.mismatch_factor = mismatch_factor(ref_twiss_zdelta,
                                               self.zdelta.twiss,
                                               transp=True)


@dataclass
class SinglePhaseSpaceBeamParameters:
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    phase_space: str

    twiss: np.ndarray | None = None

    alpha: np.ndarray | None = None
    beta: np.ndarray | None = None
    gamma: np.ndarray | None = None

    eps: np.ndarray | None = None
    envelope_pos: np.ndarray | None = None
    envelope_energy: np.ndarray | None = None

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
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
        none_to_nan : bool, optional
            To convert None to np.NaN. The default is True.
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

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        none_to_nan=False, elt=elt, pos=pos,
                                        **kwargs)

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def init_from_sigma(self, sigma: np.ndarray) -> None:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        self._compute_eps_from_sigma(sigma)
        self._compute_twiss_from_sigma(sigma)
        self.compute_envelopes()

    def init_from_another_plane(self, eps_orig: np.ndarray,
                                twiss_orig: np.ndarray, gamma_kin: np.ndarray,
                                convert: str) -> None:
        """
        Fully initialize from another phase space.

        It needs emittance and Twiss to be used, and computes envelopes. Hence
        it is not adapted to TraceWin data treatment, as we do not have the
        Twiss but already have envelopes.
        """
        self._compute_eps_from_other_plane(eps_orig, gamma_kin, convert)
        self._compute_twiss_from_other_plane(twiss_orig, gamma_kin, convert)
        self.compute_envelopes()

    def init_from_another_plane_no_twiss(self, eps_orig: np.ndarray,
                                         envelope_pos_orig: np.ndarray,
                                         envelope_energy_orig: np.ndarray,
                                         gamma_kin: np.ndarray, convert: str
                                         ) -> None:
        """
        Partially initialize from another phase space (no Twiss).

        To be used with TraceWin data treatment, as we do not need Twiss and
        already have envelopes.

        Parameters
        ----------
        eps_orig : np.ndarray
            Emittance in original phase space.
        envelope_pos_orig : np.ndarray
            Position envelope in original phase space.
        envelope_energy_orig : np.ndarray
            Energy envelope in original phase space.
        gamma_kin : np.ndarray
            Lorentz factor.
        convert : str
            To determine which phase space we have and which one we want.

        """
        self._compute_eps_from_other_plane(eps_orig, gamma_kin, convert)
        self._compute_envelopes_from_other_plane(envelope_pos_orig,
                                                 envelope_energy_orig,
                                                 gamma_kin, convert)

    def _compute_eps_from_sigma(self, sigma: np.ndarray) -> None:
        """Compute eps from sigma matrix."""
        assert self.phase_space == 'zdelta'
        self.eps = np.array(
            [np.sqrt(np.linalg.det(sigma[i])) for i in range(sigma.shape[0])])

    def _compute_eps_from_other_plane(self, eps_orig: np.ndarray,
                                      gamma_kin: np.ndarray, convert: str
                                      ) -> None:
        """Compute eps from eps in another plane."""
        self.eps = converters.emittance(eps_orig, gamma_kin, convert)

    def _compute_twiss_from_sigma(self, sigma: np.ndarray) -> None:
        """Compute the Twiss parameters using the sigma matrix."""
        assert self.eps is not None and self.phase_space == 'zdelta'
        n_points = sigma.shape[0]
        twiss = np.full((n_points, 3), np.NaN)

        for i in range(n_points):
            twiss[i, :] = np.array([-sigma[i][1, 0],
                                    sigma[i][0, 0] * 10.,
                                    sigma[i][1, 1] / 10.]) / self.eps[i]
            # beta multiplied by 10 to match TW
            # gamma divided by 10 to keep beta * gamma - alpha**2 = 1
        self._unpack_twiss(twiss)
        self.twiss = twiss

    def _compute_twiss_from_other_plane(self, twiss_orig: np.ndarray,
                                        gamma_kin: np.ndarray, convert: str
                                        ) -> None:
        """Compute Twiss parameters from Twiss parameters in another plane."""
        self.twiss = converters.twiss(twiss_orig, gamma_kin, convert)
        self._unpack_twiss(self.twiss)

    def _compute_envelopes_from_other_plane(self,
                                            envelope_pos_orig: np.ndarray,
                                            envelope_energy_orig: np.ndarray,
                                            gamma_kin: np.ndarray, convert: str
                                            ) -> None:
        """Compute envelopes from envelopes in another plane."""
        self.envelope_pos = converters.envelope_pos(envelope_pos_orig,
                                                    gamma_kin, convert)
        self.envelope_energy = converters.envelope_energy(envelope_energy_orig,
                                                          gamma_kin, convert)

    def compute_envelopes(self) -> None:
        """Compute the envelopes from the Twiss parameters and eps."""
        assert None not in (self.eps.all(), self.beta.all(), self.gamma.all())
        self.envelope_pos = np.sqrt(self.beta * self.eps)
        self.envelope_energy = np.sqrt(self.gamma * self.eps)

    def _unpack_twiss(self, twiss: np.ndarray) -> None:
        """Unpack a three-columns twiss array in alpha, beta, gamma."""
        self.alpha = twiss[:, 0]
        self.beta = twiss[:, 1]
        self.gamma = twiss[:, 2]

    def init_from_tracewin_results(self, alpha: np.ndarray, beta: np.ndarray,
                                   gamma: np.ndarray, eps: np.ndarray,
                                   envelope_pos: np.ndarray,
                                   envelope_energy: np.ndarray) -> None:
        """Init from TW."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps
        self.envelope_pos = envelope_pos
        self.envelope_energy = envelope_energy


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
           "eps_w": converters.emittance(eps_zdelta, gamma, "zdelta to w"),
           "eps_z": converters.emittance(eps_zdelta, gamma, "zdelta to z")}
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
             "twiss_w": converters.twiss(twiss_zdelta, gamma, "zdelta to w"),
             "twiss_z": converters.twiss(twiss_zdelta, gamma, "zdelta to z")}
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


def _phase_space_name_hidden_in_key(key: str) -> bool:
    """Look for the name of a phase-space in a key name."""
    if '_' not in key:
        return False

    to_test = key.split('_')
    if to_test[-1] in PHASE_SPACES:
        return True
    return False


def _separate_var_from_phase_space(key: str) -> bool:
    """Separate variable name from phase space name."""
    splitted = key.split('_')
    key = '_'.join(splitted[:-1])
    phase_space = splitted[-1]
    return key, phase_space
