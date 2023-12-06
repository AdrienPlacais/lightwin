#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle the beam parameters of a single phase space.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from dataclasses import dataclass
from typing import Self

import numpy as np

from core.beam_parameters.helper import (
    mismatch_from_arrays,
    resample_twiss_on_fix,
    sigma_from_transfer_matrices,
)
from core.beam_parameters.phase_space.i_phase_space_beam_parameters import (
    IPhaseSpaceBeamParameters,
)


@dataclass
class PhaseSpaceBeamParameters(IPhaseSpaceBeamParameters):
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    # Override some types from mother class
    eps_no_normalisation: np.ndarray
    eps_normalized: np.ndarray
    mismatch_factor: np.ndarray | None = None

    # Already with proper type in mother class:
    # envelopes: np.ndarray | None = None
    # twiss: np.ndarray | None = None
    # tm_cumul: np.ndarray | None = None
    # sigma: np.ndarray | None = None

    @classmethod
    def from_cumulated_transfer_matrices(cls,
                                         phase_space_name: str,
                                         sigma_in: np.ndarray,
                                         tm_cumul: np.ndarray,
                                         gamma_kin: np.ndarray,
                                         beta_kin: np.ndarray,
                                         ) -> Self:
        r"""Compute :math:`\sigma` matrix, and everything from it."""
        sigma = sigma_from_transfer_matrices(sigma_in, tm_cumul)
        phase_space = cls.from_sigma(phase_space_name,
                                     sigma,
                                     gamma_kin,
                                     beta_kin,
                                     tm_cumul=tm_cumul)
        return phase_space

    @classmethod
    def from_sigma(cls,
                   phase_space_name: str,
                   sigma: np.ndarray,
                   gamma_kin: np.ndarray,
                   beta_kin: np.ndarray,
                   **kwargs: np.ndarray  # tm_cumul
                   ) -> Self:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        return super().from_sigma(phase_space_name,
                                  sigma,
                                  gamma_kin,
                                  beta_kin,
                                  **kwargs)

    @classmethod
    def from_other_phase_space(cls,
                               other_phase_space: Self,
                               phase_space_name: str,
                               gamma_kin: np.ndarray,
                               beta_kin: np.ndarray,
                               **kwargs: np.ndarray,  # sigma, tm_cumul
                               ) -> Self:
        """Fully initialize from another phase space."""
        return super().from_other_phase_space(other_phase_space,
                                              phase_space_name,
                                              gamma_kin,
                                              beta_kin,
                                              **kwargs)

    @classmethod
    def from_averaging_x_and_y(cls,
                               phase_space_name: str,
                               x_space: Self,
                               y_space: Self
                               ) -> Self:
        """Create average transverse phase space from [xx'] and [yy'].

        ``eps`` is always initialized. ``mismatch_factor`` is calculated if it
        was already calculated in ``x_space`` and ``y_space``.

        """
        assert phase_space_name == 't'
        eps_normalized = .5 * (x_space.eps_normalized + y_space.eps_normalized)
        eps_no_normalisation = .5 * (x_space.eps_no_normalisation
                                     + y_space.eps_no_normalisation)
        envelopes = .5 * (x_space.envelopes + y_space.envelopes)

        mismatch_factor = None
        if x_space.mismatch_factor is not None \
                and y_space.mismatch_factor is not None:
            mismatch_factor = .5 * (x_space.mismatch_factor
                                    + y_space.mismatch_factor)

        phase_space = cls(phase_space_name=phase_space_name,
                          eps_no_normalisation=eps_no_normalisation,
                          eps_normalized=eps_normalized,
                          envelopes=envelopes,
                          mismatch_factor=mismatch_factor,
                          )
        return phase_space

    @property
    def alpha(self) -> np.ndarray | None:
        """Get first column of ``self.twiss``."""
        if self.twiss is None:
            return None
        return self.twiss[:, 0]

    @alpha.setter
    def alpha(self, value: np.ndarray) -> None:
        """Set first column of ``self.twiss``."""
        assert self.twiss is not None
        self.twiss[:, 0] = value

    @property
    def beta(self) -> np.ndarray | None:
        """Get second column of ``self.twiss``."""
        if self.twiss is None:
            return None
        return self.twiss[:, 1]

    @beta.setter
    def beta(self, value: np.ndarray) -> None:
        """Set second column of ``self.twiss``."""
        assert self.twiss is not None
        self.twiss[:, 1] = value

    @property
    def gamma(self) -> np.ndarray | None:
        """Get third column of ``self.twiss``."""
        if self.twiss is None:
            return None
        return self.twiss[:, 2]

    @gamma.setter
    def gamma(self, value: np.ndarray) -> None:
        """Set third column of ``self.twiss``."""
        assert self.twiss is not None
        self.twiss[:, 2] = value

    @property
    def envelope_pos(self) -> np.ndarray | None:
        """Get first column of ``self.envelopes``."""
        if self.envelopes is None:
            return None
        return self.envelopes[:, 0]

    @envelope_pos.setter
    def envelope_pos(self, value: np.ndarray) -> None:
        """Set first column of ``self.envelopes``."""
        assert self.envelopes is not None
        self.envelopes[:, 0] = value

    @property
    def envelope_energy(self) -> np.ndarray | None:
        """Get second column of ``self.envelopes``."""
        if self.envelopes is None:
            return None
        return self.envelopes[:, 1]

    @envelope_energy.setter
    def envelope_energy(self, value: np.ndarray) -> None:
        """Set second column of ``self.envelopes``."""
        assert self.envelopes is not None
        self.envelopes[:, 1] = value

    @property
    def eps(self) -> np.ndarray:
        """Return the normalized emittance."""
        return self.eps_normalized

    @property
    def sigma_in(self) -> np.ndarray:
        r"""Return the first :math:`\sigma` beam matrix."""
        assert self.sigma is not None
        return self.sigma[0]


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
