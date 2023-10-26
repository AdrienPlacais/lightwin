#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a function to generate a :class:`.BeamParameters` for Envelope1D."""
import numpy as np

from core.beam_parameters.beam_parameters import BeamParameters
from core.beam_parameters.factory import BeamParametersFactory
from core.transfer_matrix import TransferMatrix


class Envelope1DBeamParametersFactory(BeamParametersFactory):
    """A class holding method to generate :class:`.BeamParameters`."""

    def factory_method(
            self,
            gamma_kin: np.ndarray,
            transfer_matrix: TransferMatrix,
            ) -> BeamParameters:
        """Create the :class:`.BeamParameters` object.

        This is the actual method that creates the BeamParameters object.

        It takes in as argument everything that may change.

        """
        gamma_kin, beta_kin = self._check_and_set_gamma_beta(gamma_kin)

        beam_parameters = BeamParameters(self.z_abs,
                                         gamma_kin,
                                         beta_kin,
                                         self.element_to_index,
                                         sigma_in=self.sigma_in)

        # todo: this should be in the BeamParameters.__init__
        beam_parameters.create_phase_spaces(*self.phase_spaces)
        # warning, this method mey also require some kwargs

        phase_space_names = ('zdelta',)
        sub_transf_mat_names = ('r_zdelta',)
        transfer_matrices = transfer_matrix.get(*sub_transf_mat_names)
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
