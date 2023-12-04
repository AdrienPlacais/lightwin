#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a function to generate a :class:`.BeamParameters` for Envelope1D."""
from typing import Callable

import numpy as np

from core.beam_parameters.beam_parameters import BeamParameters
from core.beam_parameters.factory import BeamParametersFactory
from core.elements.element import Element
from core.transfer_matrix.transfer_matrix import TransferMatrix


class BeamParametersFactoryEnvelope1D(BeamParametersFactory):
    """A class holding method to generate :class:`.BeamParameters`."""

    def factory_method(
            self,
            sigma_in: np.ndarray,
            z_abs: np.ndarray,
            gamma_kin: np.ndarray,
            transfer_matrix: TransferMatrix,
            element_to_index: Callable[[str | Element, str | None],
                                       int | slice]
            ) -> BeamParameters:
        """Create the :class:`.BeamParameters` object."""
        z_abs, gamma_kin, beta_kin = self._check_and_set_arrays(z_abs,
                                                                gamma_kin)
        sigma_in = self._check_sigma_in(sigma_in)

        beam_parameters = BeamParameters(z_abs,
                                         gamma_kin,
                                         beta_kin,
                                         element_to_index,
                                         phase_spaces_names=self.phase_spaces,
                                         sigma_in=sigma_in)

        phase_space_names = ('zdelta',)
        sub_transf_mat_names = ('r_zdelta',)
        transfer_matrices = transfer_matrix.get(*sub_transf_mat_names)
        transfer_matrices = transfer_matrices,  # followig func expects a tuple
        self._set_from_transfer_matrix(beam_parameters,
                                       phase_space_names,
                                       transfer_matrices,
                                       gamma_kin,
                                       beta_kin)

        for phase_space_name in ('z', 'phiw'):
            self._convert_phase_space(beam_parameters,
                                      'zdelta',
                                      phase_space_name,
                                      gamma_kin,
                                      beta_kin)

        return beam_parameters
