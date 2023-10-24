#!/usr/bin/env python3
"""Handle the beam parameters of a single phase space.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from typing import Any, Callable
from dataclasses import dataclass
import logging

import numpy as np

import config_manager as con

from core.beam_parameters.helper import (sigma_beam_matrices,
                                         reconstruct_sigma,
                                         mismatch_from_arrays,
                                         resample_twiss_on_fix,
                                         )
from core.elements.element import Element

from util import converters
from util.helper import (recursive_items,
                         recursive_getter,
                         range_vals_object)


@dataclass
class SinglePhaseSpaceBeamParameters:
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    phase_space: str

    sigma_in: np.ndarray | None = None
    sigma: np.ndarray | None = None
    tm_cumul: np.ndarray | None = None

    twiss: np.ndarray | None = None

    alpha: np.ndarray | float | None = None
    beta: np.ndarray | float | None = None
    gamma: np.ndarray | None = None

    eps: np.ndarray | float | None = None
    envelope_pos: np.ndarray | None = None
    envelope_energy: np.ndarray | None = None

    mismatch_factor: np.ndarray | None = None

    element_to_index: Callable[[str | Element, str | None], int | slice] \
        | None = None

    def __post_init__(self):
        """Set the default attributes for the zdelta."""
        if self.phase_space == 'zdelta' and self.sigma_in is None:
            self.sigma_in = con.SIGMA_ZDELTA
        self._eps_no_norm: np.ndarray

    def __str__(self) -> str:
        """Show amplitude of some of the attributes."""
        out = f"\tSinglePhaseSpaceBeamParameters {self.phase_space}:\n"
        for key in ('alpha', 'beta', 'eps', 'envelope_pos', 'envelope_energy',
                    'mismatch_factor'):
            out += "\t\t" + range_vals_object(self, key)
        return out

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: Element | None = None, pos: str | None = None,
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
        elt : Element | None, optional
            If provided, return the attributes only at the considered Element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            Element.
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
                                        none_to_nan=False, **kwargs)

            if val[key] is None:
                continue

            if None not in (self.element_to_index, elt):
                idx = self.element_to_index(elt=elt, pos=pos)
                val[key] = val[key][idx]

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def init_from_cumulated_transfer_matrices(
            self, gamma_kin: np.ndarray, tm_cumul: np.ndarray | None = None,
            beta_kin: np.ndarray | None = None) -> None:
        """
        Use transfer matrices to compute ``sigma``, and then everything else.

        Used by the :class:`.Envelope1D` solver.

        """
        if self.tm_cumul is None and tm_cumul is None:
            logging.error("Missing `tm_cumul` to compute beam parameters.")
            return

        if self.tm_cumul is None:
            self.tm_cumul = tm_cumul
        if tm_cumul is None:
            tm_cumul = self.tm_cumul
        if beta_kin is None:
            beta_kin = converters.energy(gamma_kin, 'gamma to beta')

        self.sigma = sigma_beam_matrices(tm_cumul, self.sigma_in)
        self.init_from_sigma(gamma_kin, beta_kin)

    def init_from_sigma(self, gamma_kin: np.ndarray, beta_kin: np.ndarray,
                        sigma: np.ndarray | None = None) -> None:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        if sigma is None:
            sigma = self.sigma

        eps_no_normalisation, eps_normalized = self._compute_eps_from_sigma(
            sigma, gamma_kin, beta_kin)
        self.eps = eps_normalized
        self._eps_no_norm = eps_no_normalisation

        self._compute_twiss_from_sigma(sigma, eps_no_normalisation)
        self.envelope_pos, self.envelope_energy = \
            self._compute_envelopes_from_sigma(sigma)

    def reconstruct_full_sigma_matrix(self,
                                      sigma_00: np.ndarray,
                                      sigma_01: np.ndarray,
                                      eps: np.ndarray,
                                      eps_is_normalized: bool = True,
                                      gamma_kin: np.ndarray | None = None,
                                      beta_kin: np.ndarray | None = None,
                                      ) -> None:
        r"""
        Get :math:`\sigma` matrix from the two top components and emittance.

        Inputs are in :math:`\mathrm{mm}` and :math:`\mathrm{mrad}`, but the
        :math:`\sigma` matrix is in SI units (:math:`\mathrm{m}` and
        :math:`\mathrm{rad}`).

        See Also
        --------
        :ref:`units-label`.

        Parameters
        ----------
        sigma_00 : np.ndarray
            Top-left component of the sigma matrix in :math:`\mathrm{mm}`.
        sigma_01 : np.ndarray
            Top-right = bottom-left component of the sigma matrix in
            :math:`\mathrm{mm.mrad}`.
        eps : np.ndarray
            Emittance in :math:`\pi.\mathrm{mm.mrad}`.
        eps_is_normalized : bool, optional
            To tell if the given emittance is already normalized. The default
            is True. In this case, it is de-normalized and ``gamma_kin`` must
            be provided.
        gamma_kin : np.ndarray | None, optional
            Lorentz gamma factor. The default is None. It is mandatory to give
            it if the emittance is given unnormalized.
        beta_kin : np.ndarray | None, optional
            Lorentz beta factor. The default is None. In this case, we compute
            it from ``gamma_kin``.

        """
        n_points = eps.shape[0]
        if self.phase_space not in ('zdelta', 'x', 'y', 'x99', 'y99'):
            logging.warning("sigma reconstruction in this phase space not "
                            "tested. You'd better check the units of the "
                            "output.")
        if eps_is_normalized:
            if gamma_kin is None:
                logging.error("It is mandatory to give ``gamma_kin`` to "
                              "compute sigma matrix. Aborting calculation of "
                              "this phase space...")
                self.sigma = np.full((n_points, 2, 2), np.NaN)
                return

            if beta_kin is None:
                beta_kin = converters.energy(gamma_kin, 'gamma to beta')
            eps /= (beta_kin * gamma_kin)

        sigma = reconstruct_sigma(sigma_00, sigma_01, eps)
        if self.phase_space in ('zdelta', 'x', 'y', 'x99', 'y99'):
            sigma *= 1e-6
        self.sigma = sigma

    def init_from_another_plane(self, eps_orig: np.ndarray,
                                twiss_orig: np.ndarray, gamma_kin: np.ndarray,
                                beta_kin: np.ndarray, convert: str) -> None:
        """Fully initialize from another phase space."""
        eps_no_normalisation, eps_normalized = \
            self._compute_eps_from_other_plane(eps_orig, convert, gamma_kin,
                                               beta_kin)
        self.eps = eps_normalized
        self._compute_twiss_from_other_plane(twiss_orig, convert, gamma_kin,
                                             beta_kin)
        eps_for_envelope = eps_no_normalisation
        if self.phase_space == 'phiw':
            eps_for_envelope = eps_normalized
        self.compute_envelopes(self.twiss[:, 1], self.twiss[:, 2],
                               eps_for_envelope)

    def init_from_averaging_x_and_y(self, x_space: object, y_space: object
                                    ) -> None:
        """Create eps for an average transverse plane phase space."""
        self.eps = .5 * (x_space.eps + y_space.eps)
        if None not in (x_space.mismatch_factor, y_space.mismatch_factor):
            self.mismatch_factor = .5 * (x_space.mismatch_factor
                                         + y_space.mismatch_factor)
        self.twiss = None
        self.envelope_pos, self.envelope_energy = None, None

    def _compute_eps_from_sigma(self, sigma: np.ndarray, gamma_kin: np.ndarray,
                                beta_kin: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray]:
        """
        Compute emittance from sigma matrix.

        For the zdelta phase space:
            sigma is in SI units
            emittance is returned in pi.mm.%
        For the transverse phase spaces:
            sigma is in SI units
            emittances is returned in pi.mm.mrad

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix in SI units.
        gamma_kin : np.ndarray
            Lorentz gamma factor.
        beta_kin : np.ndarray
            Lorentz beta factor.

        Returns
        -------
        eps_no_normalisation : np.ndarray
            Emittance not normalized.
        eps_normalized : np.ndarray
            Emittance normalized.

        """
        assert self.phase_space in ['zdelta', 'x', 'y', 'x99', 'y99']
        eps_no_normalisation = np.array(
            [np.sqrt(np.linalg.det(sigma[i])) for i in range(sigma.shape[0])])

        if self.phase_space in ['zdelta']:
            eps_no_normalisation *= 1e5
        elif self.phase_space in ['x', 'y', 'x99', 'y99']:
            eps_no_normalisation *= 1e6

        eps_normalized = converters.emittance(eps_no_normalisation,
                                              f"normalize {self.phase_space}",
                                              gamma_kin=gamma_kin,
                                              beta_kin=beta_kin)
        return eps_no_normalisation, eps_normalized

    def _compute_eps_from_other_plane(self, eps_orig: np.ndarray, convert: str,
                                      gamma_kin: np.ndarray,
                                      beta_kin: np.ndarray
                                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert emittance from another phase space.

        Output emittance is normalized if input is, and is un-normalized if the
        input emittance is not normalized.

        Parameters
        ----------
        eps_orig : np.ndarray
            Emittance of starting phase-space.
        convert : str
            To tell nature of starting and ending phase spaces.
        gamma_kin : np.ndarray | None
            Lorentz gamma.
        beta_kin : np.ndarray | None
            Lorentz  beta

        Returns
        -------
        eps_new : np.ndarray
            Emittance in the new phase-space, with the same normalisation state
            as eps_orig.

        """
        eps_normalized = converters.emittance(eps_orig, convert,
                                              gamma_kin=gamma_kin,
                                              beta_kin=beta_kin)

        eps_no_normalisation = converters.emittance(
            eps_normalized,
            f"de-normalize {self.phase_space}",
            gamma_kin,
            beta_kin
        )
        return eps_no_normalisation, eps_normalized

    def _compute_twiss_from_sigma(self, sigma: np.ndarray,
                                  eps_no_normalisation: np.ndarray
                                  ) -> None:
        """
        Compute the Twiss parameters using the sigma matrix.

        For the zdelta phase space:
            sigma is in SI units
            eps_no_normalisation should be in pi.mm.%
        For the transverse planes:
            sigma is in SI units
            eps_no_normalisation is in pi.mm.mrad

        Parameters
        ----------
        sigma : np.ndarray
            sigma matrix along the linac.
        eps_no_normalisation : np.ndarray
            Unnormalized emittance.

        Returns
        -------
        Nothing, but set attributes alpha (no units), beta (mm/pi.%), gamma
        (mm.pi/%) and twiss (column_stacking of the three), in the z-dp/p
        plane.
        In the transverse planes, units are pi mm and mrad instead.

        """
        assert self.phase_space in ['zdelta', 'x', 'y', 'x99', 'y99']
        assert self.eps is not None
        n_points = sigma.shape[0]
        twiss = np.full((n_points, 3), np.NaN)

        for i in range(n_points):
            twiss[i, :] = np.array(
                [-sigma[i][1, 0], sigma[i][0, 0], sigma[i][1, 1]]
            ) / eps_no_normalisation[i] * 1e6

        if self.phase_space == 'zdelta':
            twiss[:, 0] *= 1e-1
            twiss[:, 2] *= 1e-2

        self._unpack_twiss(twiss)
        self.twiss = twiss

    def _compute_twiss_from_other_plane(self, twiss_orig: np.ndarray,
                                        convert: str, gamma_kin: np.ndarray,
                                        beta_kin: np.ndarray) -> None:
        """Compute Twiss parameters from Twiss parameters in another plane."""
        self.twiss = converters.twiss(twiss_orig, gamma_kin, convert,
                                      beta_kin=beta_kin)
        self._unpack_twiss(self.twiss)

    # TODO would be possible to skip this with TW, where envelope_pos is
    # already known
    def _compute_envelopes_from_sigma(self, sigma: np.ndarray
                                      ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the envelopes.

        Units are mm for the position envelope in [zdelta], [x-x'], [y-y'].

        Energy envelope:
            [zdelta]: %
            [x-x'], [y-y']: mrad

        """
        envelope_pos = np.array([np.sqrt(sigm[0, 0]) for sigm in sigma]) * 1e3
        envelope_energy = np.array([np.sqrt(sigm[1, 1]) for sigm in sigma]
                                   ) * 1e3

        if self.phase_space == 'zdelta':
            envelope_energy /= 10.

        return envelope_pos, envelope_energy

    def compute_envelopes(self, beta: np.ndarray, gamma: np.ndarray,
                          eps: np.ndarray) -> None:
        """
        Compute the envelopes from the Twiss parameters and eps.

        Emittance eps should be normalized in the [phi-W] plane, but not in the
        [z-delta] and [z-z'] planes (consistency with TW).

        """
        self.envelope_pos = np.sqrt(beta * eps)
        self.envelope_energy = np.sqrt(gamma * eps)

    def _unpack_twiss(self, twiss: np.ndarray) -> None:
        """Unpack a three-columns twiss array in alpha, beta, gamma."""
        self.alpha = twiss[:, 0]
        self.beta = twiss[:, 1]
        self.gamma = twiss[:, 2]


def mismatch_single_phase_space(ref: SinglePhaseSpaceBeamParameters,
                                fix: SinglePhaseSpaceBeamParameters,
                                z_ref: np.ndarray,
                                z_fix: np.ndarray
                                ) -> np.ndarray | None:
    """Compute mismatch between two :class:`SinglePhaseSpaceBeamParameters`."""
    twiss_ref, twiss_fix = ref.twiss, fix.twiss
    assert twiss_ref is not None and twiss_fix is not None
    if twiss_ref.shape != twiss_fix.shape:
        twiss_ref = resample_twiss_on_fix(z_ref, twiss_ref, z_fix)

    mism = mismatch_from_arrays(twiss_ref, twiss_fix, transp=True)
    return mism
