#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hold the beam parameters in a single phase space."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np

from core.beam_parameters.helper import (
    envelopes_from_sigma,
    envelopes_from_twiss_eps,
    eps_from_other_phase_space,
    eps_from_sigma,
    twiss_from_other_phase_space,
    twiss_from_sigma,
)
from util.helper import range_vals_object

IMPLEMENTED_PHASE_SPACES = (
    "zdelta",
    "z",
    "phiw",
    "x",
    "y",
    "t",
    "phiw99",
    "x99",
    "y99",
)  #:


@dataclass
class IPhaseSpaceBeamParameters(ABC):
    """Hold Twiss, emittance, envelopes of single phase-space @ single pos."""

    phase_space_name: str
    eps_no_normalisation: np.ndarray | float
    eps_normalized: np.ndarray | float
    envelopes: np.ndarray | None = None
    twiss: np.ndarray | None = None
    sigma: np.ndarray | None = None
    tm_cumul: np.ndarray | None = None
    mismatch_factor: np.ndarray | float | None = None

    def __post_init__(self) -> None:
        """Ensure that the phase space exists."""
        assert self.phase_space_name in IMPLEMENTED_PHASE_SPACES

    @classmethod
    def from_sigma(
        cls,
        phase_space_name: str,
        sigma: np.ndarray,
        gamma_kin: np.ndarray | float,
        beta_kin: np.ndarray | float,
        **kwargs: np.ndarray,
    ) -> Self:
        """Compute Twiss, eps, envelopes just from sigma matrix."""
        eps_no_normalisation, eps_normalized = eps_from_sigma(
            phase_space_name,
            sigma,
            gamma_kin,
            beta_kin,
        )
        twiss = twiss_from_sigma(phase_space_name, sigma, eps_no_normalisation)
        envelopes = envelopes_from_sigma(phase_space_name, sigma)
        phase_space = cls(
            phase_space_name=phase_space_name,
            eps_no_normalisation=eps_no_normalisation,
            eps_normalized=eps_normalized,
            sigma=sigma,
            twiss=twiss,
            envelopes=envelopes,
            **kwargs,
        )
        return phase_space

    @classmethod
    def from_other_phase_space(
        cls,
        other_phase_space: Self,
        phase_space_name: str,
        gamma_kin: np.ndarray | float,
        beta_kin: np.ndarray | float,
        **kwargs: np.ndarray,  # sigma, tm_cumul
    ) -> Self:
        """Fully initialize from another phase space."""
        other_phase_space_name = other_phase_space.phase_space_name
        eps_other = other_phase_space.eps_normalized
        twiss_other = other_phase_space.twiss
        assert twiss_other is not None

        eps_no_normalisation, eps_normalized = eps_from_other_phase_space(
            other_phase_space_name,
            phase_space_name,
            eps_other,
            gamma_kin,
            beta_kin,
        )
        twiss = twiss_from_other_phase_space(
            other_phase_space_name,
            phase_space_name,
            twiss_other,
            gamma_kin,
            beta_kin,
        )

        eps_for_envelope = eps_no_normalisation
        if phase_space_name == "phiw":
            eps_for_envelope = eps_normalized
        envelopes = envelopes_from_twiss_eps(twiss, eps_for_envelope)
        phase_space = cls(
            phase_space_name=phase_space_name,
            eps_no_normalisation=eps_no_normalisation,
            eps_normalized=eps_normalized,
            twiss=twiss,
            envelopes=envelopes,
            **kwargs,
        )
        return phase_space

    def __str__(self) -> str:
        """Show amplitude of some of the attributes."""
        out = f"\t\tPhase space {self.phase_space_name}:\n"
        for key in (
            "alpha",
            "beta",
            "eps",
            "envelope_pos",
            "envelope_energy",
            "mismatch_factor",
        ):
            out += "\t\t\t" + range_vals_object(self, key)
        return out

    def __repr__(self) -> str:
        """Give same information as str."""
        return self.__str__()

    @property
    @abstractmethod
    def alpha(self) -> np.ndarray | float | None:
        """Get first element/column of ``self.twiss``."""
        pass

    @alpha.setter
    @abstractmethod
    def alpha(self, value: np.ndarray | float) -> None:
        """Set first element/column of ``self.twiss``."""
        pass

    @property
    @abstractmethod
    def beta(self) -> np.ndarray | float | None:
        """Get second element/column of ``self.twiss``."""
        pass

    @beta.setter
    @abstractmethod
    def beta(self, value: np.ndarray | float) -> None:
        """Set second element/column of ``self.twiss``."""
        pass

    @property
    @abstractmethod
    def gamma(self) -> np.ndarray | float | None:
        """Get third element/column of ``self.twiss``."""
        pass

    @gamma.setter
    @abstractmethod
    def gamma(self, value: np.ndarray | float) -> None:
        """Set third element/column of ``self.twiss``."""
        pass

    @property
    @abstractmethod
    def envelope_pos(self) -> np.ndarray | float | None:
        """Get first element/column of ``self.envelopes``."""
        pass

    @envelope_pos.setter
    @abstractmethod
    def envelope_pos(self, value: np.ndarray | float) -> None:
        """Set first element/column of ``self.envelopes``."""
        pass

    @property
    @abstractmethod
    def envelope_energy(self) -> np.ndarray | float | None:
        """Get second element/column of ``self.envelopes``."""
        pass

    @envelope_energy.setter
    @abstractmethod
    def envelope_energy(self, value: np.ndarray | float) -> None:
        """Set second element/column of ``self.envelopes``."""
        pass

    @property
    @abstractmethod
    def eps(self) -> np.ndarray | float:
        """Return the normalized emittance."""
        pass
