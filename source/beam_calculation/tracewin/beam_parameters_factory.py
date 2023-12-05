#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a function to generate a :class:`.BeamParameters` for TraceWin."""
from typing import Callable

import numpy as np

from core.beam_parameters.beam_parameters import BeamParameters
from core.beam_parameters.factory import BeamParametersFactory
from core.beam_parameters.helper import reconstruct_sigma
from core.elements.element import Element


class BeamParametersFactoryTraceWin(BeamParametersFactory):
    """A class holding method to generate :class:`.BeamParameters`."""

    def factory_method(
            self,
            z_abs: np.ndarray,
            gamma_kin: np.ndarray,
            results: dict[str, np.ndarray],
            element_to_index: Callable[[str | Element, str | None],
                                       int | slice],
    ) -> BeamParameters:
        """Create the :class:`.BeamParameters` object."""
        z_abs, gamma_kin, beta_kin = self._check_and_set_arrays(z_abs,
                                                                gamma_kin)

        beam_parameters = BeamParameters(z_abs,
                                         gamma_kin,
                                         beta_kin,
                                         sigma_in=None,
                                         element_to_index=element_to_index,
                                         )

        phase_space_names = ('x', 'y', 'zdelta')
        data_to_retrieve_sigmas = (
            self._extract_phase_space_data_for_sigma(phase_space_name,
                                                     results)
            for phase_space_name in phase_space_names
        )
        sigmas = (
            reconstruct_sigma(phase_space_name,
                              *data_to_retrieve_sigma,
                              eps_is_normalized=True,
                              gamma_kin=gamma_kin,
                              beta_kin=beta_kin,
                              )
            for phase_space_name, data_to_retrieve_sigma
            in zip(phase_space_names, data_to_retrieve_sigmas)
        )
        self._set_from_sigma(beam_parameters,
                             phase_space_names,
                             sigmas,
                             gamma_kin,
                             beta_kin,
                             )

        other_phase_space_names = ('x', 'y')
        phase_space_name = 't'
        self._set_transverse_from_x_and_y(beam_parameters,
                                          other_phase_space_names,
                                          phase_space_name)

        other_phase_space_name = 'zdelta'
        phase_space_names = ('z', 'phiw')
        self._set_from_other_phase_space(beam_parameters,
                                         other_phase_space_name,
                                         phase_space_names,
                                         gamma_kin,
                                         beta_kin)

        if self.is_multipart:
            phase_space_names = ('x99', 'y99', 'phiw99')
            emittances = (
                self._extract_emittance_for_99percent(phase_space_name,
                                                      results)
                for phase_space_name in phase_space_names
            )
            self._set_only_emittance(beam_parameters,
                                     phase_space_names,
                                     emittances)
        return beam_parameters

    def _extract_phase_space_data_for_sigma(
            self,
            phase_space_name: str,
            results: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r"""
        Retrieve the data necessary to reconstruct :math:`\sigma` beam matrix.

        Parameters
        ----------
        phase_space_name : {'x', 'y', 'zdelta'}
            Name of a single phase space.
        results : dict[str, np.ndarray]
            Results dictionary, which keys are ``tracewin.out`` or
            ``partran1.out`` headers and which values are corresponding data.

        Returns
        -------
        sigma_00 : np.ndarray
            ``(n, )`` array containing top-left component of the :math:`\sigma`
            beam matrix.
        sigma_01 : np.ndarray
            ``(n, )`` array containing top-right component of the
            :math:`\sigma` beam matrix.
        eps_normalized : np.ndarray
            ``(n, )`` array of normalized emittance.

        """
        phase_space_to_keys = {
            'x': ('SizeX', "sxx'", 'ex'),
            'y': ('SizeY', "syy'", 'ey'),
            'zdelta': ('SizeZ', "szdp", 'ezdp'),
        }
        assert phase_space_name in phase_space_to_keys
        keys = phase_space_to_keys[phase_space_name]
        sigma_00 = results[keys[0]]**2
        sigma_01 = results[keys[1]]
        eps_normalized = results[keys[2]]
        return sigma_00, sigma_01, eps_normalized

    def _extract_emittance_for_99percent(self,
                                         phase_space_name: str,
                                         results: dict[str, np.ndarray],
                                         ) -> np.ndarray:
        r"""
        Retrieve the 99% emittances.

        .. todo::
            normalized or not???

        Parameters
        ----------
        phase_space_name : {'x99', 'y99', 'phiw99'}
            Name of a single phase space.
        results : dict[str, np.ndarray]
            Results dictionary, which keys are ``tracewin.out`` or
            ``partran1.out`` headers and which values are corresponding data.

        Returns
        -------
        eps : np.ndarray
            99% emittance in the desired phase space.

        """
        getters = {
            'x99': results['ex99'],
            'y99': results['ey99'],
            'phiw99': results['ep99']
        }
        assert phase_space_name in getters
        return getters[phase_space_name]
