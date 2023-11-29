#!/usr/bin/env python3
"""Handle the beam parameters of a single phase space.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from typing import Any, Callable, Self
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
from util.helper import range_vals_object


IMPLEMENTED_PHASE_SPACES = ('zdelta', 'z', 'phiw', 'x', 'y', 't',
                            'phiw99', 'x99', 'y99')  #:
BEAM_PARAMETERS = ('sigma_in',
                   'sigma',
                   'tm_cumul',
                   'twiss', 'alpha', 'beta', 'gamma',
                   'eps',
                   'envelope',
                   'mismatch_factor'
                   )  #:


class PhaseSpaceBeamParameters:
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    def __init__(self,
                 phase_space: str,
                 n_points: int,
                 element_to_index: Callable[[str | Element, str | None],
                                            int | slice] | None = None,
                 **kwargs: np.ndarray | None
                 ) -> None:
        """Instantiate the objects with given or fallback values.

        Parameters
        ----------
        phase_space : str
            Name of the phase space. Check :data:`IMPLEMENTED_PHASE_SPACES` for
            allowed values.
        element_to_index : Callable[[str | Element, str | None],
                                     int | slice] | None, optional
            Takes an :class:`.Element`, its name, ``'first'`` or ``'last'`` as
            argument, and returns corresponding index. Index should be the same
            in all the arrays attributes of this class: ``z_abs``,
            ``beam_parameters`` attributes, etc. Used to easily ``get`` the
            desired properties at the proper position. The default is None.
        n_points : int
            Number of points of the arrays. Used to set default empty arrays.
        kwargs : np.ndarray | None
            Dict which keys must be in :data:`BEAM_PARAMETERS`, which is
            checked by :meth:`._check_input_is_implemented`. Data is saved as
            attribute if it is an array, else a default array full of np.NaN is
            saved. This is handled by :meth:`._store_input_if_not_none`.

        """
        assert phase_space in IMPLEMENTED_PHASE_SPACES
        self.phase_space = phase_space
        self.element_to_index = element_to_index
        self.n_points = n_points

        self._check_input_is_implemented(tuple(kwargs.keys()))
        self.sigma_in: np.ndarray
        self.sigma: np.ndarray
        self.tm_cumul: np.ndarray
        self.twiss: np.ndarray
        self.eps: np.ndarray
        self.envelope: np.ndarray
        self.mismatch_factor: np.ndarray
        self._store_input_if_not_none(**kwargs)

        self._eps_no_norm: np.ndarray

    def _check_input_is_implemented(self, keys_of_kwargs: tuple[str, ...]
                                    ) -> None:
        """Verify that the keys of ``kwargs`` match an implemented quantity.

        Parameters
        ----------
        keys_of_kwargs : tuple[str, ...]
            Keys of the ``kwargs`` given to :meth:`.__init__`.

        """
        for key in keys_of_kwargs:
            if key in BEAM_PARAMETERS:
                continue

            logging.warning(f"You tried to initialize a beam parameter, {key},"
                            f" which is not supported. Ignoring this key...")

    def _store_input_if_not_none(self, **kwargs: np.ndarray | None) -> None:
        """Save the data from ``kwargs`` as attribute if they are not None.

        If they are None, we initialize it with an array full of np.NaN which
        has the proper dimensions.

        Parameters
        ----------
        kwargs : np.ndarray | None
            Keys must be in ``BEAM_PARAMETERS``, which is verified by
            :meth:`._check_input_is_implemented`.

        """
        defaults = {
            'sigma_in': np.full((2, 2), np.NaN),
            'sigma': np.full((self.n_points, 2, 2), np.NaN),
            'tm_cumul': np.full((self.n_points, 2, 2), np.NaN),
            'twiss': np.full((self.n_points, 3), np.NaN),
            'eps': np.full((self.n_points), np.NaN),
            'envelope': np.full((self.n_points, 2), np.NaN),
            'mismatch_factor': np.full((self.n_points), np.NaN),
        }
        for attribute_name, default_value in defaults.items():
            given_value = kwargs.get(attribute_name, None)
            if given_value is None:
                setattr(self, attribute_name, default_value)
                continue

            setattr(self, attribute_name, given_value)

        aliases = ('alpha', 'beta', 'gamma', 'envelope_pos', 'envelope_energy')
        for attribute_name, given_value in kwargs.items():
            if attribute_name in aliases and given_value is not None:
                setattr(self, attribute_name, given_value)

    def is_not_set(self, name: str) -> bool:
        """Tells if there is a np.Nan in the array."""
        array = self.get(name)
        assert isinstance(array, np.ndarray)
        return np.isnan(array).any()

    def is_set(self, name: str) -> bool:
        """Tells if there is no np.Nan in the array."""
        return ~self.is_not_set(name)

    @property
    def alpha(self) -> np.ndarray:
        """Get first column of ``self.twiss``."""
        return self.twiss[:, 0]

    @alpha.setter
    def alpha(self, value: np.ndarray | float) -> None:
        """Set first column of ``self.twiss``."""
        self.twiss[:, 0] = np.atleast_1d(value)

    @property
    def beta(self) -> np.ndarray:
        """Get second column of ``self.twiss``."""
        return self.twiss[:, 1]

    @beta.setter
    def beta(self, value: np.ndarray | float) -> None:
        """Set second column of ``self.twiss``."""
        self.twiss[:, 1] = np.atleast_1d(value)

    @property
    def gamma(self) -> np.ndarray:
        """Get third column of ``self.twiss``."""
        return self.twiss[:, 2]

    @gamma.setter
    def gamma(self, value: np.ndarray | float) -> None:
        """Set third column of ``self.twiss``."""
        self.twiss[:, 2] = np.atleast_1d(value)

    @property
    def envelope_pos(self) -> np.ndarray:
        """Get first column of ``self.envelope``."""
        return self.envelope[:, 0]

    @envelope_pos.setter
    def envelope_pos(self, value: np.ndarray | float) -> None:
        """Set first column of ``self.envelope``."""
        self.envelope[:, 0] = np.atleast_1d(value)

    @property
    def envelope_energy(self) -> np.ndarray:
        """Get second column of ``self.envelope``."""
        return self.envelope[:, 1]

    @envelope_energy.setter
    def envelope_energy(self, value: np.ndarray) -> None:
        """Set second column of ``self.envelope``."""
        self.envelope[:, 1] = np.atleast_1d(value)

    def __str__(self) -> str:
        """Show amplitude of some of the attributes."""
        out = f"\tPhaseSpaceBeamParameters {self.phase_space}:\n"
        for key in ('alpha', 'beta', 'eps', 'envelope_pos', 'envelope_energy',
                    'mismatch_factor'):
            out += "\t\t" + range_vals_object(self, key)
        return out

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        # return key in recursive_items(vars(self))
        return hasattr(self, key)

    def get(self,
            *keys: str,
            elt: Element | None = None,
            pos: str | None = None,
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
            val[key] = getattr(self, key)

        if elt is not None:
            assert self.element_to_index is not None
            idx = self.element_to_index(elt=elt, pos=pos)
            val = {_key: _value[idx] for _key, _value in val.items()}

        if len(keys) == 1:
            return val[keys[0]]

        out = [val[key] for key in keys]
        return tuple(out)

    def init_from_cumulated_transfer_matrices(self,
                                              tm_cumul: np.ndarray,
                                              gamma_kin: np.ndarray,
                                              beta_kin: np.ndarray,
                                              ) -> None:
        r"""Compute :math:`\sigma` matrix, and everything from it."""
        if tm_cumul is None:
            logging.error("tm_cumul shall be given!")
            tm_cumul = self.tm_cumul

        assert self.is_set('sigma_in')
        self.sigma = sigma_beam_matrices(tm_cumul, self.sigma_in)
        self.init_from_sigma(self.sigma, gamma_kin, beta_kin)

    def init_from_sigma(self,
                        sigma: np.ndarray,
                        gamma_kin: np.ndarray,
                        beta_kin: np.ndarray,
                        ) -> None:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        self.sigma_in = sigma[0]
        eps_no_normalisation, eps_normalized = self._compute_eps_from_sigma(
            sigma,
            gamma_kin,
            beta_kin)
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
        Set :math:`\sigma` matrix from the two top components and emittance.

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
        if self.phase_space not in ('zdelta', 'x', 'y', 'x99', 'y99'):
            logging.warning("sigma reconstruction in this phase space not "
                            "tested. You'd better check the units of the "
                            "output.")
        if eps_is_normalized:
            if gamma_kin is None:
                logging.error("It is mandatory to give ``gamma_kin`` to "
                              "compute sigma matrix. Aborting calculation of "
                              "this phase space...")
                return

            if beta_kin is None:
                beta_kin = converters.energy(gamma_kin, 'gamma to beta')
            eps /= (beta_kin * gamma_kin)

        sigma = reconstruct_sigma(sigma_00, sigma_01, eps)
        if self.phase_space in ('zdelta', 'x', 'y', 'x99', 'y99'):
            sigma *= 1e-6
        assert isinstance(sigma, np.ndarray)
        self.sigma = sigma
        self.sigma_in = sigma[0]

    def init_from_another_plane(self,
                                eps_orig: np.ndarray,
                                twiss_orig: np.ndarray,
                                gamma_kin: np.ndarray,
                                beta_kin: np.ndarray,
                                convert: str) -> None:
        """Fully initialize from another phase space."""
        eps_no_normalisation, eps_normalized = \
            self._compute_eps_from_other_plane(eps_orig,
                                               convert,
                                               gamma_kin,
                                               beta_kin)
        self.eps = eps_normalized
        self._compute_twiss_from_other_plane(twiss_orig,
                                             convert,
                                             gamma_kin,
                                             beta_kin)
        eps_for_envelope = eps_no_normalisation
        if self.phase_space == 'phiw':
            eps_for_envelope = eps_normalized

        assert self.is_set('twiss')
        self.compute_envelopes(self.twiss[:, 1],
                               self.twiss[:, 2],
                               eps_for_envelope)

    def init_from_averaging_x_and_y(self,
                                    x_space: Self,
                                    y_space: Self
                                    ) -> None:
        """Create average transverse phase space from [xx'] and [yy'].

        ``eps`` is always initialized. ``mismatch_factor`` is calculated if it
        was already calculated in ``x_space`` and ``y_space``.

        """
        assert x_space.is_set('eps')
        assert y_space.is_set('eps')
        self.eps = .5 * (x_space.eps + y_space.eps)

        if x_space.is_set('mismatch_factor') and \
                y_space.is_set('mismatch_factor'):
            self.mismatch_factor = .5 * (x_space.mismatch_factor
                                         + y_space.mismatch_factor)

    def _compute_eps_from_sigma(self,
                                sigma: np.ndarray,
                                gamma_kin: np.ndarray,
                                beta_kin: np.ndarray,
                                replace_nan_by_0: bool = True,
                                ) -> tuple[np.ndarray, np.ndarray]:
        r"""
        Compute emittance from :math:`\sigma` beam matrix.

        In the :math:`[z-\delta]` phase space, emittance is in
        :math:`\pi.\mathrm{mm.\%}`.
        In the transverse phase spaces, emittance is in
        :math:`\pi.\mathrm{mm.mrad}`.
        :math:`\sigma` is always in SI units.

        Parameters
        ----------
        sigma : np.ndarray
            :math:`\sigma` beam matrix in SI units.
        gamma_kin : np.ndarray
            Lorentz gamma factor.
        beta_kin : np.ndarray
            Lorentz beta factor.
        replace_nan_by_0 : bool, optional
            To avoid raising of errors and allow the code to reach the creation
            of a :class:`.SimulationOutput`. Useful for debugging.

        Returns
        -------
        eps_no_normalisation : np.ndarray
            Emittance not normalized.
        eps_normalized : np.ndarray
            Emittance normalized.

        """
        allowed = ('zdelta', 'x', 'y', 'x99', 'y99')
        assert self.phase_space in allowed, \
            f"Phase-space {self.phase_space} not in {allowed}."

        eps_no_normalisation = np.array(
            [np.sqrt(np.linalg.det(sigma[i])) for i in range(sigma.shape[0])])

        if self.phase_space in ('zdelta'):
            eps_no_normalisation *= 1e5
        elif self.phase_space in ('x', 'y', 'x99', 'y99'):
            eps_no_normalisation *= 1e6

        if replace_nan_by_0 and np.isnan(eps_no_normalisation).any():
            logging.error("Replacing NaN by 0. in emittance array.")
            eps_no_normalisation[np.where(np.isnan(eps_no_normalisation))] = 0.

        assert ~np.isnan(eps_no_normalisation).any()
        eps_normalized = converters.emittance(eps_no_normalisation,
                                              f"normalize {self.phase_space}",
                                              gamma_kin=gamma_kin,
                                              beta_kin=beta_kin)
        assert ~np.isnan(eps_normalized).any()
        return eps_no_normalisation, eps_normalized

    def _compute_eps_from_other_plane(self,
                                      eps_orig: np.ndarray,
                                      convert: str,
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

    def _compute_twiss_from_sigma(self,
                                  sigma: np.ndarray,
                                  eps_no_normalisation: np.ndarray
                                  ) -> None:
        r"""Compute the Twiss parameters using the :math:`\sigma` matrix.

        In the :math:`[z-\delta]` phase space, emittance and Twiss are in
        :math:`\mathrm{mm}` and :math:`\mathrm{\%}`.
        In the transverse phase spaces, emittance and Twiss are in
        :math:`\mathrm{mm}` and :math:`\mathrm{mrad}`.
        :math:`\sigma` is always in SI units.

        .. todo::
            Would be better if all emittances had the same units? Check
            consistency with rest of the code...

        .. todo::
            Check if property setter work with the *= thingy

        Parameters
        ----------
        sigma : np.ndarray
            (n, 2, 2) array holding :math:`\sigma` beam matrix.
        eps_no_normalisation : np.ndarray
            Unnormalized emittance.

        """
        assert self.phase_space in ('zdelta', 'x', 'y', 'x99', 'y99')
        assert self.is_set('eps')
        n_points = sigma.shape[0]
        twiss = np.full((n_points, 3), np.NaN)

        for i in range(n_points):
            twiss[i, :] = np.array(
                [-sigma[i][1, 0], sigma[i][0, 0], sigma[i][1, 1]]
            ) / eps_no_normalisation[i] * 1e6

        if self.phase_space == 'zdelta':
            twiss[:, 0] *= 1e-1
            twiss[:, 2] *= 1e-2

        self.twiss = twiss

    def _compute_twiss_from_other_plane(self,
                                        twiss_orig: np.ndarray,
                                        convert: str,
                                        gamma_kin: np.ndarray,
                                        beta_kin: np.ndarray) -> None:
        """Compute Twiss parameters from Twiss parameters in another plane.

        Parameters
        ----------
        twiss_orig : np.ndarray
            (n, 3) Twiss array from original phase space.
        convert : str
            Tells the original and destination phase spaces. Format is
            ``'{original} to {destination}'``.
        gamma_kin : np.ndarray
            Array of gamma Lorentz factor.
        beta_kin : np.ndarray
            Array of beta Lorentz factor.

        See Also
        --------
        :func:`helper.converters.twiss` for list of allowed values for
        ``convert``.

        """
        self.twiss = converters.twiss(twiss_orig,
                                      gamma_kin,
                                      convert,
                                      beta_kin=beta_kin)

    # TODO would be possible to skip this with TW, where envelope_pos is
    # already known
    def _compute_envelopes_from_sigma(self,
                                      sigma: np.ndarray
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


def mismatch_single_phase_space(ref: PhaseSpaceBeamParameters,
                                fix: PhaseSpaceBeamParameters,
                                z_ref: np.ndarray,
                                z_fix: np.ndarray
                                ) -> np.ndarray | None:
    """Compute mismatch between two :class:`PhaseSpaceBeamParameters`."""
    twiss_ref, twiss_fix = ref.twiss, fix.twiss
    assert twiss_ref is not None and twiss_fix is not None
    if twiss_ref.shape != twiss_fix.shape:
        twiss_ref = resample_twiss_on_fix(z_ref, twiss_ref, z_fix)

    mism = mismatch_from_arrays(twiss_ref, twiss_fix, transp=True)
    return mism
