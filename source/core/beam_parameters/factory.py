#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a factory for the :class:`.BeamParameters`."""
from abc import ABC, abstractmethod
import logging
from typing import Iterable, Sequence

import numpy as np

from beam_calculation.simulation_output.simulation_output import SimulationOutput
from core.elements.element import Element
from util import converters

from .beam_parameters import BeamParameters
from .initial_beam_parameters import InitialBeamParameters
from .phase_space.initial_phase_space_beam_parameters import (
    InitialPhaseSpaceBeamParameters,
)
from .phase_space.phase_space_beam_parameters import PhaseSpaceBeamParameters


# Subclassed for every BeamCalculator
class BeamParametersFactory(ABC):
    """
    Declare factory method, that returns the :class:`.BeamParameters`.

    Subclassed by every :class:`.BeamCalculator`.

    """

    def __init__(self,
                 is_3d: bool,
                 is_multipart: bool,
                 ) -> None:
        """Initialize the class."""
        self.phase_spaces = self._determine_phase_spaces(is_3d, is_multipart)
        self.is_3d = is_3d
        self.is_multipart = is_multipart

    def _determine_phase_spaces(self,
                                is_3d: bool,
                                is_multipart: bool) -> tuple[str, ...]:
        if not is_3d:
            return ('z', 'zdelta', 'phiw')
        if not is_multipart:
            return ('x', 'y', 't', 'z', 'zdelta', 'phiw')
        return ('x', 'y', 't', 'z', 'zdelta', 'phiw', 'x99', 'y99', 'phiw99')

    @abstractmethod
    def factory_method(self, *args, **kwargs) -> BeamParameters:
        """Create the :class:`.BeamParameters` object."""
        beam_parameters = BeamParameters(*args, **kwargs)
        return beam_parameters

    def _check_and_set_arrays(
            self,
            z_abs: np.ndarray | float,
            gamma_kin: np.ndarray | float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Ensure that inputs are arrays with proper shape, compute beta."""
        z_abs = np.atleast_1d(z_abs)
        gamma_kin = np.atleast_1d(gamma_kin)
        assert gamma_kin.shape == z_abs.shape, "Shape mismatch: " \
            + f"{gamma_kin.shape = } different from {z_abs.shape = }."

        beta_kin = converters.energy(gamma_kin, 'gamma to beta')
        assert isinstance(beta_kin, np.ndarray)
        return z_abs, gamma_kin, beta_kin

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

    def _set_from_other_phase_space(self,
                                    beam_parameters: BeamParameters,
                                    other_phase_space_name: str,
                                    phase_space_names: Sequence[str],
                                    gamma_kin: np.ndarray,
                                    beta_kin: np.ndarray) -> None:
        """Instantiate a phase space from another one.

        Parameters
        ----------
        beam_parameters : BeamParameters
            Object holding the beam parameters in different phase spaces.
        other_phase_space_name : {'zdelta'}
            Name of the phase space from which the new phase space will be
            initialized.
        phase_space_names : Sequence[{'phiw', 'z'}]
            Name of the phase spaces that will be created.
        gamma_kin : np.ndarray
            Lorentz gamma factor.
        beta_kin : np.ndarray
            Lorentz beta factor.

        """
        implemented_in = ('zdelta', )
        assert other_phase_space_name in implemented_in, \
            f"{other_phase_space_name = } not in {implemented_in = }"
        other_phase_space = beam_parameters.get(other_phase_space_name)

        implemented_out = ('phiw', 'z')
        for phase_space_name in phase_space_names:
            assert phase_space_name in implemented_out, \
                f"{phase_space_name = } not in {implemented_out = }"

            phase_space = PhaseSpaceBeamParameters.from_other_phase_space(
                other_phase_space,
                phase_space_name,
                gamma_kin,
                beta_kin,
            )
            setattr(beam_parameters, phase_space_name, phase_space)

    def _set_only_emittance(self,
                            beam_parameters: BeamParameters,
                            phase_space_names: Sequence[str],
                            emittances: Iterable[np.ndarray]) -> None:
        """Set only the emittance."""
        for phase_space_name, eps in zip(phase_space_names, emittances):
            phase_space = PhaseSpaceBeamParameters(phase_space_name,
                                                   eps_no_normalisation=eps,
                                                   eps_normalized=eps,
                                                   )
            setattr(beam_parameters, phase_space_name, phase_space)

    def _set_from_transfer_matrix(self,
                                  beam_parameters: BeamParameters,
                                  phase_space_names: Sequence[str],
                                  transfer_matrices: Sequence[np.ndarray],
                                  gamma_kin: np.ndarray,
                                  beta_kin: np.ndarray
                                  ) -> None:
        """Initialize phase spaces from their transfer matrices.

        Parameters
        ----------
        beam_parameters : BeamParameters
            Object holding the different phase spaces.
        phase_space_names : Sequence[str]
            Names of the phase spaces to initialize.
        transfer_matrices : Sequence[np.ndarray]
            Transfer matrix corresponding to each phase space.
        gamma_kin : np.ndarray
            Lorentz gamma factor.
        beta_kin : np.ndarray
            Lorentz beta factor.

        """
        for phase_space_name, transfer_matrix in zip(phase_space_names,
                                                     transfer_matrices):
            sigma_in = beam_parameters.sub_sigma_in(phase_space_name)
            phase_space = \
                PhaseSpaceBeamParameters.from_cumulated_transfer_matrices(
                    phase_space_name=phase_space_name,
                    sigma_in=sigma_in,
                    tm_cumul=transfer_matrix,
                    gamma_kin=gamma_kin,
                    beta_kin=beta_kin)
            setattr(beam_parameters, phase_space_name, phase_space)

    def _set_transverse_from_x_and_y(self,
                                     beam_parameters: BeamParameters,
                                     other_phase_space_names: tuple[str, str],
                                     phase_space_name: str,
                                     ) -> None:
        """Initialize ``t`` (transverse) phase space.

        Parameters
        ----------
        beam_parameters : BeamParameters
            Object already holding the beam parameters in the ``x`` and ``y``
            phase spaces.

        """
        x_space = getattr(beam_parameters, other_phase_space_names[0])
        y_space = getattr(beam_parameters, other_phase_space_names[1])
        phase_space = PhaseSpaceBeamParameters.from_averaging_x_and_y(
            phase_space_name,
            x_space,
            y_space)
        setattr(beam_parameters, 't', phase_space)

    def _set_from_sigma(self,
                        beam_parameters: BeamParameters,
                        phase_space_names: Sequence[str],
                        sigmas: Iterable[np.ndarray],
                        gamma_kin: np.ndarray,
                        beta_kin: np.ndarray,
                        ) -> None:
        r"""Initialize transfer matrices from :math:`\sigma` beam matrix."""
        for phase_space_name, sigma in zip(phase_space_names, sigmas):
            phase_space = PhaseSpaceBeamParameters.from_sigma(
                phase_space_name,
                sigma,
                gamma_kin,
                beta_kin,
            )
            setattr(beam_parameters, phase_space_name, phase_space)


# Subclassed by ListOfElements
# (for now, ListOfElements is common to every BeamCalculator)
class InitialBeamParametersFactory(ABC):
    """
    This is used when creating new :class:`.ListOfElements`.

    This factory is not subclassed. Only one instance should be created.

    .. todo::
        Remove the ``is_3d``, ``is_multipart`` as I always create the same
        object with ``True``, ``True``.
        Also :meth:`self._determine_phase_spaces` is useless.

    """

    def __init__(self,
                 is_3d: bool,
                 is_multipart: bool) -> None:
        """Create factory and list of phase spaces to generate.

        Parameters
        ----------
        is_3d : bool
            If the simulation is in 3D.
        is_multipart : bool
            If the simulation is a multiparticle.

        """
        # self.phase_spaces = self._determine_phase_spaces(is_3d)
        # self.is_3d = is_3d
        # self.is_multipart = is_multipart

        self.phase_spaces = ('x', 'y', 'z', 'zdelta')

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

        input_beam = InitialBeamParameters(z_abs, gamma_kin, beta_kin)

        phase_space_names = ('x', 'y', 'zdelta')
        sigmas = (sigma_in[:2, :2], sigma_in[2:4, 2:4], sigma_in[4:, 4:])
        self._set_from_sigma(input_beam, phase_space_names, sigmas)

        other_phase_space_name = 'zdelta'
        phase_space_names = ('z', )
        self._set_from_other_phase_space(input_beam,
                                         other_phase_space_name,
                                         phase_space_names)
        return input_beam

    def factory_subset(self,
                       simulation_output: SimulationOutput,
                       get_kw: dict[str, Element | str | bool | None],
                       ) -> InitialBeamParameters:
        """Generate :class:`.InitialBeamParameters` for a linac portion.

        Parameters
        ----------
        simulation_output : SimulationOutput
            Object from which the beam parameters data will be taken.
        get_kw : dict[str, Element | str | bool | None]
            dict that can be passed to the `get` method and that will return
            the data at the beginning of the linac portion.

        Returns
        -------
        InitialBeamParameters
            Holds information on the beam at the beginning of the linac
            portion.

        """
        beam_parameters_kw = self._initial_beam_parameters_kw(
            simulation_output,
            get_kw)
        input_beam = InitialBeamParameters(**beam_parameters_kw)

        original_beam_parameters = simulation_output.beam_parameters
        assert original_beam_parameters is not None

        phase_space_names = self.phase_spaces
        skip_missing_phase_spaces = True
        input_phase_spaces_kw = self._initial_phase_space_beam_parameters_kw(
            original_beam_parameters,
            phase_space_names,
            get_kw,
            skip_missing_phase_spaces)
        for key, value in input_phase_spaces_kw.items():
            phase_space = InitialPhaseSpaceBeamParameters(phase_space_name=key,
                                                          **value)
            setattr(input_beam, key, phase_space)

        other_phase_space_name = 'zdelta'
        phase_space_names = ('z', )
        self._set_from_other_phase_space(input_beam,
                                         other_phase_space_name,
                                         phase_space_names)
        return input_beam

    def _initial_beam_parameters_kw(
        self,
        simulation_output: SimulationOutput,
        get_kw: dict[str, Element | str | bool | None],
    ) -> dict[str, float]:
        """Generate the kw to instantiate the :class:`.InitialBeamParameters`.

        Parameters
        ----------
        simulation_output : SimulationOutput
            Object from which the initial beam will be taken.
        get_kw : dict[str, Element | str | bool | None]
            Keyword argument to ``get`` ``args`` at proper position.

        Returns
        -------
        dict[str, float]
            Dictionary of keyword arguments for
            :class:`.InitialBeamParameters`.

        """
        args = ('z_abs', 'gamma', 'beta')
        z_abs, gamma, beta = simulation_output.get(*args, **get_kw)
        beam_parameters_kw = {
            'z_abs': z_abs,
            'gamma_kin': gamma,
            'beta_kin': beta,
        }
        return beam_parameters_kw

    def _initial_phase_space_beam_parameters_kw(
            self,
            original_beam_parameters: BeamParameters,
            phase_space_names: Sequence[str],
            get_kw: dict[str, Element | str | bool | None],
            skip_missing_phase_spaces: bool,
    ) -> dict[str, dict[str, float | np.ndarray]]:
        """Get all beam data at proper position and store it in a dict.

        Parameters
        ----------
        original_beam_parameters : BeamParameters
            Object holding original beam parameters.
        get_kw : dict[str, Element | str | bool | None]
            dict that can be passed to the `get` method and that will return
            the data at the beginning of the linac portion.
        skip_missing_phase_spaces : bool
            To handle when a phase space from :var:`self.phase_spaces` is not
            defined in ``original_beam_parameters``, and is therefore not
            initializable. If True, we just skip it. If False and such a case
            happens, an ``AttributeError`` will be raised.

        Returns
        -------
        initial_phase_spaces_kw : dict[str, dict[str, float | np.ndarray]]
            Keys are the name of the phase spaces.
            The values are other dictionaries, which keys-values are
            :class:`.PhaseSpaceInitialBeamParameters` attributes.

        """
        args = ('eps_no_normalisation',
                'eps_normalized',
                'envelopes',
                'twiss',
                'tm_cumul',
                'sigma',
                )
        to_skip = (skip_missing_phase_spaces
                   and not hasattr(original_beam_parameters, phase_space_name)
                   for phase_space_name in phase_space_names)

        initial_phase_spaces_kw = {}
        for phase_space_name, to_skip in zip(phase_space_names, to_skip):
            if to_skip:
                continue

            initial_phase_space_kw = {
                key: original_beam_parameters.get(
                    key,
                    phase_space_name=phase_space_name,
                    **get_kw)
                for key in args
            }
            initial_phase_spaces_kw[phase_space_name] = initial_phase_space_kw

        return initial_phase_spaces_kw

    def _set_from_sigma(self,
                        initial_beam_parameters: InitialBeamParameters,
                        phase_space_names: Sequence[str],
                        sigmas: Iterable[np.ndarray],
                        ) -> None:
        r"""Initialize transfer matrices from :math:`\sigma` beam matrix."""
        for phase_space_name, sigma in zip(phase_space_names, sigmas):
            phase_space = InitialPhaseSpaceBeamParameters.from_sigma(
                phase_space_name,
                sigma,
                initial_beam_parameters.gamma_kin,
                initial_beam_parameters.beta_kin,
            )
            setattr(initial_beam_parameters, phase_space_name, phase_space)

    def _set_from_other_phase_space(
            self,
            initial_beam_parameters: InitialBeamParameters,
            other_phase_space_name: str,
            phase_space_names: Sequence[str],
    ) -> None:
        """Instantiate a phase space from another one.

        Parameters
        ----------
        initial_beam_parameters : InitialBeamParameters
            Object holding the beam parameters in different phase spaces.
        other_phase_space_name : {'zdelta'}
            Name of the phase space from which the new phase space will be
            initialized.
        phase_space_names : Sequence[{'phiw', 'z'}]
            Name of the phase spaces that will be created.
        gamma_kin : float
            Lorentz gamma factor.
        beta_kin : float
            Lorentz beta factor.

        """
        implemented_in = ('zdelta', )
        assert other_phase_space_name in implemented_in, \
            f"{other_phase_space_name = } not in {implemented_in = }"
        other_phase_space = initial_beam_parameters.get(other_phase_space_name)

        implemented_out = ('phiw', 'z')
        for phase_space_name in phase_space_names:
            assert phase_space_name in implemented_out, \
                f"{phase_space_name = } not in {implemented_out = }"

            phase_space = InitialPhaseSpaceBeamParameters.\
                from_other_phase_space(
                    other_phase_space,
                    phase_space_name,
                    initial_beam_parameters.gamma_kin,
                    initial_beam_parameters.beta_kin
                )
            setattr(initial_beam_parameters, phase_space_name, phase_space)
