#!/usr/bin/env python3
"""Handle the initial beam parameters of a single phase space.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from typing import Any
import logging
from dataclasses import dataclass

import numpy as np

from core.beam_parameters.helper import (sigma_beam_matrices,
                                         reconstruct_sigma,
                                         mismatch_from_arrays,
                                         resample_twiss_on_fix,
                                         )
from core.elements.element import Element

from util import converters
from util.dicts_output import markdown


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


@dataclass
class PhaseSpaceInitialBeamParameters:
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    phase_space: str
    eps: float | None = None
    twiss: np.ndarray | None = None
    sigma: np.ndarray | None = None
    envelope: np.ndarray | None = None
    tm_cumul: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Legacy."""
        self._eps_no_norm: np.ndarray
        self._none_to_nan_arrays()

    def _none_to_nan_arrays(self) -> None:
        defaults = {
            'sigma': np.full((2, 2), np.NaN),
            'tm_cumul': np.full((2, 2), np.NaN),
            'twiss': np.full(3, np.NaN),
            'envelope': np.full(2, np.NaN),
        }
        for attribute_name, default_value in defaults.items():
            if self.get(attribute_name) is None:
                setattr(self, attribute_name, default_value)

    # legacy
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

    # legacy
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
            'twiss': np.full(3, np.NaN),
            'eps': np.NaN,
            'envelope': np.full(2, np.NaN),
            'mismatch_factor': np.NaN,
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
    def alpha(self) -> float:
        """Get first item of ``self.twiss``."""
        return self.twiss[0]

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set first item of ``self.twiss``."""
        self.twiss[0] = value

    @property
    def beta(self) -> float:
        """Get second item of ``self.twiss``."""
        return self.twiss[1]

    @beta.setter
    def beta(self, value: float) -> None:
        """Set second item of ``self.twiss``."""
        self.twiss[1] = value

    @property
    def gamma(self) -> float:
        """Get third item of ``self.twiss``."""
        return self.twiss[2]

    @gamma.setter
    def gamma(self, value: float) -> None:
        """Set third item of ``self.twiss``."""
        self.twiss[2] = value

    @property
    def envelope_pos(self) -> float:
        """Get first item of ``self.envelope``."""
        return self.envelope[0]

    @envelope_pos.setter
    def envelope_pos(self, value: float) -> None:
        """Set first item of ``self.envelope``."""
        self.envelope[0] = value

    @property
    def envelope_energy(self) -> float:
        """Get second item of ``self.envelope``."""
        return self.envelope[1]

    @envelope_energy.setter
    def envelope_energy(self, value: float) -> None:
        """Set second item of ``self.envelope``."""
        self.envelope[1] = value

    @property
    def sigma_in(self) -> np.ndarray:
        """Set an alias for sigma."""
        assert self.sigma is not None
        return self.sigma

    def __str__(self) -> str:
        """Show amplitude of some of the attributes."""
        out = f"\tPhaseSpaceBeamParameters {self.phase_space}:\n"
        for key in ('alpha', 'beta', 'eps', 'envelope_pos', 'envelope_energy',
                    'mismatch_factor'):
            out += f"\t\t{markdown[key]} {self.get(key)}"
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

    def init_from_sigma(self,
                        gamma_kin: float,
                        beta_kin: float,
                        ) -> None:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        eps_no_normalisation, eps_normalized = self._compute_eps_from_sigma(
            self.sigma,
            gamma_kin,
            beta_kin)
        self.eps = eps_normalized
        self._eps_no_norm = eps_no_normalisation

        self._compute_twiss_from_sigma(self.sigma, eps_no_normalisation)
        self.envelope_pos, self.envelope_energy = \
            self._compute_envelopes_from_sigma(self.sigma)

    def init_eye_tm_cumul(self) -> None:
        """Set eye transfer matrix."""
        self.tm_cumul = np.ones((2, 2))

    def _compute_eps_from_sigma(self,
                                sigma: np.ndarray,
                                gamma_kin: float,
                                beta_kin: float
                                ) -> tuple[float, float]:
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
        gamma_kin : float
            Lorentz gamma factor.
        beta_kin : float
            Lorentz beta factor.

        Returns
        -------
        eps_no_normalisation : float
            Emittance not normalized.
        eps_normalized : float
            Emittance normalized.

        """
        allowed = ('zdelta', 'x', 'y', 'x99', 'y99')
        assert self.phase_space in allowed, \
            f"Phase-space {self.phase_space} not in {allowed}."

        eps_no_normalisation = np.sqrt(np.linalg.det(sigma))

        if self.phase_space in ('zdelta'):
            eps_no_normalisation *= 1e5
        elif self.phase_space in ('x', 'y', 'x99', 'y99'):
            eps_no_normalisation *= 1e6

        eps_normalized = converters.emittance(eps_no_normalisation,
                                              f"normalize {self.phase_space}",
                                              gamma_kin=gamma_kin,
                                              beta_kin=beta_kin)
        assert isinstance(eps_normalized, float)
        return eps_no_normalisation, eps_normalized

    def _compute_twiss_from_sigma(self,
                                  sigma: np.ndarray,
                                  eps_no_normalisation: float
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
        eps_no_normalisation : float
            Unnormalized emittance.

        """
        assert self.phase_space in ('zdelta', 'x', 'y', 'x99', 'y99')
        twiss = np.array([-sigma[1, 0], sigma[0, 0], sigma[1, 1]])
        twiss /= eps_no_normalisation
        twiss *= 1e6

        if self.phase_space == 'zdelta':
            twiss[0] *= 1e-1
            twiss[2] *= 1e-2
        self.twiss = twiss

    # TODO would be possible to skip this with TW, where envelope_pos is
    # already known
    def _compute_envelopes_from_sigma(self,
                                      sigma: np.ndarray
                                      ) -> tuple[float, float]:
        """
        Compute the envelopes.

        Units are mm for the position envelope in [zdelta], [x-x'], [y-y'].

        Energy envelope:
            [zdelta]: %
            [x-x'], [y-y']: mrad

        """
        envelope_pos = np.sqrt(sigma[0, 0]) * 1e3
        envelope_energy = np.sqrt(sigma[1, 1]) * 1e3
        if self.phase_space == 'zdelta':
            envelope_energy /= 10.
        return envelope_pos, envelope_energy
