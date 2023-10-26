#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a function to generate a :class:`.BeamParameters` for TraceWin."""
from typing import Callable
import logging

import numpy as np

from core.elements.element import Element
from core.beam_parameters.beam_parameters import BeamParameters
from core.beam_parameters.factory import BeamParametersFactory

from util import converters


class TraceWinBeamParametersFactory(BeamParametersFactory):
    """A class holding method to generate :class:`.BeamParameters`."""

    def factory_method(
            self,
            gamma_kin: np.ndarray,
            results: dict[str, np.ndarray],
            ) -> BeamParameters:
        """Create the :class:`.BeamParameters` object.

        This is the actual method that creates the BeamParameters object.

        It takes in as argument everything that may change.

        """
        gamma_kin, beta_kin = self._check_and_set_gamma_beta(gamma_kin)

        beam_parameters = BeamParameters(self.z_abs,
                                         gamma_kin,
                                         beta_kin,
                                         self.element_to_index)

        # todo: this should be in the BeamParameters.__init__
        beam_parameters.create_phase_spaces(*self.phase_spaces)
        # warning, this method mey also require some kwargs

        for phase_space_name in ('x', 'y', 'zdelta'):
            self._set_everything_from_sigma(beam_parameters,
                                            phase_space_name,
                                            results,
                                            gamma_kin,
                                            beta_kin)

        beam_parameters.t.init_from_averaging_x_and_y(beam_parameters.x,
                                                      beam_parameters.y)

        for phase_space_name in ('z', 'phiw'):
            self._convert_phase_space(beam_parameters,
                                      'zdelta',
                                      phase_space_name,
                                      gamma_kin,
                                      beta_kin)

        if self.is_multipart:
            phase_space_names = ('x99', 'y99', 'phi99')
            self._set_99percent_emittances(beam_parameters,
                                           results,
                                           *phase_space_names)
        return beam_parameters

    def _set_everything_from_sigma(self,
                                   beam_parameters: BeamParameters,
                                   phase_space_name: str,
                                   results: dict[str, np.ndarray],
                                   gamma_kin: np.ndarray,
                                   beta_kin: np.ndarray,
                                   ) -> None:
        args = self._extract_phase_space_data_for_sigma(phase_space_name,
                                                        results)
        phase_space = beam_parameters.get(phase_space_name)
        phase_space.reconstruct_full_sigma_matrix(*args,
                                                  eps_is_normalized=True,
                                                  gamma_kin=gamma_kin,
                                                  beta_kin=beta_kin)
        phase_space.init_from_sigma(phase_space.sigma,
                                    gamma_kin,
                                    beta_kin)

    def _extract_phase_space_data_for_sigma(
            self,
            phase_space_name: str,
            results: dict[str, np.ndarray],
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        phase_space_to_keys = {
            'x': ('SizeX', "sxx'", 'ex'),
            'y': ('SizeY', "syy'", 'ey'),
            'zdelta': ('SizeZ', "szdp", 'ezdp'),
        }
        keys = phase_space_to_keys[phase_space_name]
        sigma_00 = results[keys[0]]**2
        sigma_01 = results[keys[1]]
        eps_normalized = results[keys[2]]
        return sigma_00, sigma_01, eps_normalized

    def _set_99percent_emittances(self,
                                  beam_parameters: BeamParameters,
                                  results: dict[str, np.ndarray],
                                  *phase_space_names: str
                                  ) -> None:
        getters = {
            'x99': results['ex99'],
            'y99': results['ey99'],
            'phiw99': results['ep99']
            }
        for phase_space_name in phase_space_names:
            assert phase_space_name in getters
            self._set_only_emittance(beam_parameters,
                                     phase_space_name,
                                     getters[phase_space_name])
