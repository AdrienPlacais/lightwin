#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a factory for the :class:`.BeamParameters`.

try something...

InitialBeamParameters in ListOfElements  -> called once per ListOfElements
BeamParameters in BeamCalculator         -> called for every SimulationOutput

... even for ListOfElements, should be linked with BeamCalculator, they do not
need the same BeamParameters!


"""
from abc import ABC, abstractmethod
from typing import Callable, Sequence
import logging

import numpy as np

from core.elements.element import Element
from core.beam_parameters.beam_parameters import BeamParameters
from core.beam_parameters.initial_beam_parameters import InitialBeamParameters
from beam_calculation.output import SimulationOutput
from util import converters


class BeamParametersFactory(ABC):
    """Declare factory method, that returns the :class:`.BeamParameters`."""

    def __init__(self,
                 z_abs: np.ndarray | float,
                 is_3d: bool,
                 is_multipart: bool,
                 element_to_index: Callable[[str | Element, str | None],
                                            int | slice] | None = None,
                 sigma_in: np.ndarray | None = None,
                 ) -> None:
        """Initialize the class.

        Here we set the things that do not vary between two executions of the
        code.

        For non-TraceWin, also provide sigma_in


        """
        self.z_abs = np.atleast_1d(z_abs)
        self.n_points = self.z_abs.shape[0]
        self.element_to_index = element_to_index

        self.phase_spaces = self._determine_phase_spaces(is_3d, is_multipart)
        self.is_3d = is_3d
        self.is_multipart = is_multipart

        if sigma_in is not None:
            self.sigma_in = sigma_in

    def _determine_phase_spaces(self,
                                is_3d: bool,
                                is_multipart: bool) -> tuple[str, ...]:
        if not is_3d:
            return ('z', 'zdelta', 'phiw')
        if not is_multipart:
            return ('x', 'y', 't', 'z', 'zdelta', 'phiw')
        return ('x', 'y', 't', 'z', 'zdelta', 'phiw', 'x99', 'y99', 'phiw99')

    @abstractmethod
    def factory_method(
            self,
            gamma_kin: np.ndarray | float,
            # sigma_in for non-tracewin
            # results dict for tracewin
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
        # warning, this method also requires some kwargs
        return beam_parameters

    def _check_and_set_gamma_beta(self, gamma_kin: np.ndarray | float
                                  ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Lorentz beta, ensure shape of arrays are consistent."""
        gamma_kin = np.atleast_1d(gamma_kin)
        assert gamma_kin.shape == self.z_abs.shape, "The shape of gamma kin "\
            f"is {gamma_kin.shape}, while it should be {self.z_abs.shape}. "\
            "Maybe the meshing or the export of the BeamCalculator slightly "\
            "changed? Try to recreate a new BeamParametersFactory whenever "\
            "this error is raised."

        beta_kin = converters.energy(gamma_kin, 'gamma to beta')
        assert isinstance(beta_kin, np.ndarray)

        return gamma_kin, beta_kin

    def _check_sigma_in(self, sigma_in: np.ndarray) -> np.ndarray:
        """Change shape of ``sigma_in`` if necessary."""
        if sigma_in.shape == (2, 2):
            assert self.is_3d, "(2, 2) shape is only for 1D simulation (and "\
                "is to be avoided)."

            logging.warning("Would be better to feed in a (6, 6) array with "
                            "NaN.")
            return sigma_in

        if sigma_in.shape == (6, 6):
            return sigma_in

        raise IOError(f"{sigma_in.shape = } not recognized.")

    def _convert_phase_space(self,
                             beam_parameters: BeamParameters,
                             phase_space_in: str,
                             phase_space_out: str,
                             gamma_kin: np.ndarray,
                             beta_kin: np.ndarray) -> None:
        """Convert one phase space to another.

        Parameters
        ----------
        beam_parameters : BeamParameters
            beam_parameters
        phase_space_in : str
            phase_space_in
        phase_space_out : str
            phase_space_out
        gamma_kin : np.ndarray
            gamma_kin
        beta_kin : np.ndarray
            beta_kin

        """
        implemented_in = ('zdelta', )
        implemented_out = ('phiw', 'z')
        assert phase_space_in in implemented_in, f"{phase_space_in = } not in"\
            f"{implemented_in = }"
        assert phase_space_out in implemented_out, f"{phase_space_out = } "\
            f"not in {implemented_out = }"

        space_in = beam_parameters.get(phase_space_in)
        space_out = beam_parameters.get(phase_space_out)
        conversion_name = f"{phase_space_in} to {phase_space_out}"
        space_out.init_from_another_plane(space_in.eps,
                                          space_in.twiss,
                                          gamma_kin,
                                          beta_kin,
                                          conversion_name)

    def _set_only_emittance(self,
                            beam_parameters: BeamParameters,
                            phase_space_name: str,
                            eps: np.ndarray) -> None:
        """Set only the emittance."""
        beam_parameters.get(phase_space_name).eps = eps

    def _set_from_transfer_matrix(self,
                                  beam_parameters: BeamParameters,
                                  phase_space_names: Sequence[str],
                                  transfer_matrices: Sequence[np.ndarray],
                                  gamma_kin: np.ndarray,
                                  beta_kin: np.ndarray
                                  ) -> None:
        for phase_space_name, transfer_matrix in zip(phase_space_names,
                                                     transfer_matrices):
            phase_space = beam_parameters.get(phase_space_name)
            phase_space.init_from_cumulated_transfer_matrices(
                tm_cumul=transfer_matrix,
                gamma_kin=gamma_kin,
                beta_kin=beta_kin
                )


class InitialBeamParametersFactory(ABC):
    """This used when creating new :class:`.ListOfElements`."""

    def __init__(self,
                 is_3d: bool,
                 is_multipart: bool) -> None:
        self.phase_spaces = self._determine_phase_spaces(is_3d, is_multipart)
        self.is_3d = is_3d
        self.is_multipart = is_multipart

    def _determine_phase_spaces(self, is_3d: bool) -> tuple[str, ...]:
        """Set the phase spaces that we will need to start a calculation.

        Parameters
        ----------
        is_3d : bool
            If the simulation is in 3D.

        Returns
        -------
        tuple[str, ...]
            Name of the phase spaces to create.

        """
        if not is_3d:
            return ('zdelta',)
        return ('x', 'y', 'zdelta')

    def factory_new(self,
                    sigma_in: np.ndarray,
                    w_kin: float,
                    z_abs: float = 0.) -> InitialBeamParameters:
        r"""Create the beam parameters for the beginning of the linac.

        Parameters
        ----------
        sigma_in : np.ndarray
            :math:`\sigma` beam matrix.
        w_kin : float
            Kinetic energy in MeV.
        z_abs : float, optional
            Absolute position of the linac start. Should be 0, which is the
            default.

        Returns
        -------
        InitialBeamParameters
            Beam parameters at the start of the linac.

        """
        gamma_kin = converters.energy(w_kin, 'kin to gamma')
        beta_kin = converters.energy(gamma_kin, 'gamma to beta')
        assert isinstance(gamma_kin, float)
        assert isinstance(beta_kin, float)
        input_beam = InitialBeamParameters(self.phase_spaces,
                                           z_abs,
                                           gamma_kin,
                                           beta_kin,
                                           )
        sub_sigmas = self._generate_dict_of_sigma_matrices(sigma_in)
        input_beam.init_phase_spaces_from_sigma(sub_sigmas)
        return input_beam

    def _generate_dict_of_sigma_matrices(self, sigma_in: np.ndarray
                                         ) -> dict[str, np.ndarray]:
        r"""Separate the sigma beam matrices into its three components.

        Parameters
        ----------
        sigma_in : np.ndarray
            (6, 6) sigma beam matrix.

        Returns
        -------
        dict[str, np.ndarray]
            Keys are name of the phase space, values are (2, 2) sigma matrices
            in corresponding plane.

        """
        sub_sigmas = {
            'x': sigma_in[:2, :2],
            'y': sigma_in[2:4, 2:4],
            'zdelta': sigma_in[4:, 4:]
            }
        return sub_sigmas

    def factory_subset(self,
                       simulation_output: SimulationOutput,
                       get_kwargs: dict[str, Element | str | bool | None],
                       ) -> InitialBeamParameters:
        """Generate :class:`.InitialBeamParameters` for a linac portion.

        Parameters
        ----------
        simulation_output : SimulationOutput
            Object from which the beam parameters data will be taken.
        get_kwargs : dict[str, Element | str | bool | None]
            dict that can be passed to the `get` method and that will return
            the data at the beginning of the linac portion.

        Returns
        -------
        InitialBeamParameters
            Holds information on the beam at the beginning of the linac
            portion.

        """
        args = ('z_abs', 'gamma', 'beta', 'sigma')
        z_abs, gamma_kin, beta_kin, sigma_in = simulation_output.get(
            *args,
            **get_kwargs)

        original_beam_parameters = simulation_output.beam_parameters
        assert original_beam_parameters is not None
        necessary = ('eps', 'twiss', 'sigma', 'envelope', 'tm_cumul')
        initial_beam_kwargs = self._generate_initial_beam_kwargs(
            original_beam_parameters,
            necessary,
            get_kwargs)

        input_beam = InitialBeamParameters(self.phase_spaces,
                                           z_abs,
                                           gamma_kin,
                                           beta_kin,)
                                           # sigma_in=np.array(sigma_in))
        input_beam.init_phase_spaces_from_kwargs(initial_beam_kwargs)
        return input_beam

    def _generate_initial_beam_kwargs(
            self,
            original_beam_parameters: BeamParameters,
            necessary: tuple[str, ...],
            get_kwargs: dict[str, Element | str | bool | None],
            ) -> dict[str, dict[str, float | np.ndarray]]:
        """Get all beam data at proper position and store it in a dict.

        Parameters
        ----------
        original_beam_parameters : BeamParameters
            Object holding original beam parameters.
        necessary : tuple[str, ...]
            Quantities necessary to initialize the
            :class:`.PhaseSpaceInitialBeamParameters`.
        get_kwargs : dict[str, Element | str | bool | None]
            dict that can be passed to the `get` method and that will return
            the data at the beginning of the linac portion.

        Returns
        -------
        initial_beam_kwargs : dict[str, dict[str, float | np.ndarray]]
            Keys are the name of the phase spaces.
            The values are other dictionaries, which keys-values are
            :class:`.PhaseSpaceInitialBeamParameters` attributes.

        """
        initial_beam_kwargs = {
            phase_space_name: self._generate_phase_space_initial_beam_kwargs(
                phase_space_name,
                original_beam_parameters,
                necessary,
                get_kwargs)
            for phase_space_name in self.phase_spaces}
        return initial_beam_kwargs

    def _generate_phase_space_initial_beam_kwargs(
            self,
            phase_space_name: str,
            original_beam_parameters: BeamParameters,
            necessary: tuple[str, ...],
            get_kwargs: dict
            ) -> dict[str, float | np.ndarray]:
        """Get beam data of a single phase space.

        Parameters
        ----------
        phase_space_name : str
            Name of the phase space.
        original_beam_parameters : BeamParameters
            Object holding original beam parameters.
        necessary : tuple[str, ...]
            Quantities necessary to initialize the
            :class:`.PhaseSpaceInitialBeamParameters`.
        get_kwargs : dict[str, Element | str | bool | None]
            dict that can be passed to the `get` method and that will return
            the data at the beginning of the linac portion.

        Returns
        -------
        phase_space_kwargs : dict[str, float | np.ndarray]
            A dict, which keys-values are
            :class:`.PhaseSpaceInitialBeamParameters` attributes.

        """
        phase_space_kwargs = {
            quantity: original_beam_parameters.get(
                quantity,
                phase_space=phase_space_name,
                **get_kwargs)
            for quantity in necessary}
        return phase_space_kwargs
