#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Hold everything related to emittances, Twiss, envelopes.

They are stored in a :class:`SinglePhaseSpaceBeamParameters`, which are
gathered in a :class:`BeamParameters` object.

.. todo::
    May be interesting to create a ``BeamInitialState``, in the same fashion as
    :class:`.ParticleInitialState`. ``BeamInitialState`` would be the only one
    to have a ``tracewin_command``.

For a list of the units associated with every parameter, see
:ref:`units-beam-parameters-label`.

"""
from typing import Any, Callable
from dataclasses import dataclass
import logging

import numpy as np

import config_manager as con

from core.elements.element import Element

from tracewin_utils.interface import beam_parameters_to_command

from util import converters
from util.helper import (recursive_items,
                         recursive_getter,
                         range_vals,
                         range_vals_object)


PHASE_SPACES = ('zdelta', 'z', 'phiw', 'x', 'y', 't',
                'phiw99', 'x99', 'y99')


@dataclass
class BeamParameters:
    """
    Hold all emittances, envelopes, etc in various planes.

    Attributes
    ----------
    z_abs : np.ndarray | None, optional
        Absolute position in the linac in m. The default is None.
    gamma_kin : np.ndarray | float | None, optional
        Lorentz gamma factor. The default is None.
    beta_kin : np.ndarray | float | None, optional
        Lorentz gamma factor. The default is None. If `beta_kin` is not
        provided but `gamma_kin` is, `beta_kin` is automatically calculated at
        initialisation.
    element_to_index : Callable[[str | Element, str | None],
                                 int | slice] | None, optional
        Takes an `Element`, its name, 'first' or 'last' as argument, and
        returns correspondinf index. Index should be the same in all the arrays
        attributes of this class: `z_abs`, `beam_parameters` attributes, etc.
        Used to easily `get` the desired properties at the proper position. The
        default is None.
    zdelta : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [z-zdelta] plane.
    z : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [z-z'] plane.
    phiw : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [phi-W] plane.
    x : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [x-x'] plane. Only used with 3D
        simulations.
    y : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [y-y'] plane. Only used with 3D
        simulations.
    t : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the [t-t'] (transverse) plane. Only used with
        3D simulations.
    phiw99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the [phi-W] plane. Only used with
        multiparticle simulations.
    x99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the [x-x'] plane. Only used with
        multiparticle simulations.
    y99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the [y-y'] plane. Only used with
        multiparticle simulations.

    """

    z_abs: np.ndarray | float | None = None
    gamma_kin: np.ndarray | float | None = None
    beta_kin: np.ndarray | float | None = None
    element_to_index: Callable[[str | Element, str | None], int | slice] \
        | None = None

    def __post_init__(self) -> None:
        """Define the attributes that may be used."""
        if self.beta_kin is None and self.gamma_kin is not None:
            self.beta_kin = converters.energy(self.gamma_kin, 'gamma to beta')

        self.zdelta: SinglePhaseSpaceBeamParameters
        self.z: SinglePhaseSpaceBeamParameters
        self.phiw: SinglePhaseSpaceBeamParameters
        self.x: SinglePhaseSpaceBeamParameters
        self.y: SinglePhaseSpaceBeamParameters
        self.t: SinglePhaseSpaceBeamParameters
        self.phiw99: SinglePhaseSpaceBeamParameters
        self.x99: SinglePhaseSpaceBeamParameters
        self.y99: SinglePhaseSpaceBeamParameters

    def __str__(self) -> str:
        """Give compact information on the data that is stored."""
        out = "\tBeamParameters:\n"
        out += "\t\t" + range_vals("zdelta.eps", self.zdelta.eps)
        out += "\t\t" + range_vals("zdelta.beta", self.zdelta.beta)
        out += "\t\t" + range_vals("zdelta.mismatch",
                                   self.zdelta.mismatch_factor)
        return out

    def has(self, key: str) -> bool:
        """
        Tell if the required attribute is in this class.

        Specifics of this method: twiss_zdelta will return True, even if the
        correct property is zdelta.twiss.
        """
        if _phase_space_name_hidden_in_key(key):
            key, phase_space = _separate_var_from_phase_space(key)
            return key in recursive_items(vars(vars(self)[phase_space]))
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: Element | None = None, pos: str | None = None,
            phase_space: str | None = None, **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        What is particular in this getter is that all
        SinglePhaseSpaceBeamParameters attributes have attributes with the same
        name: `twiss`, `alpha`, `beta`, `gamma`, `eps`, `envelopes_pos` and
        `envelopes_energy`.
        Hence, you must provide either a `phase_space` argument which shall be
        in PHASE_SPACES, either you must append the name of the phase space to
        the name of the desired variable.

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
        phase_space : ['z', 'zdelta', 'phi_w', 'x', 'y'] | None, optional
            Phase space in which you want the key. The default is None. In this
            case, the quantities from the zdelta phase space are taken.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}
        for key in keys:
            short_key = key
            if _phase_space_name_hidden_in_key(key):
                if phase_space is not None:
                    logging.warning(
                        "Amibiguous: you asked for two phase-spaces. One with "
                        f"keyword argument {phase_space = }, and another with "
                        f"the positional argument {key = }. I take phase "
                        f"space from {key = }.")
                short_key, phase_space = _separate_var_from_phase_space(key)

            if not self.has(short_key):
                val[key] = None
                continue

            for stored_key, stored_val in vars(self).items():
                if stored_key == short_key:
                    val[key] = stored_val

                    if None not in (self.element_to_index, elt):
                        idx = self.element_to_index(elt=elt, pos=pos)
                        val[key] = val[key][idx]

                    break

                if stored_key == phase_space:
                    val[key] = recursive_getter(
                        short_key, vars(stored_val), to_numpy=False,
                        none_to_nan=False, **kwargs)

                    if val[key] is None:
                        continue

                    if None not in (self.element_to_index, elt):
                        idx = self.element_to_index(elt=elt, pos=pos)
                        val[key] = val[key][idx]

                    break

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    @property
    def tracewin_command(self) -> list[str]:
        """Return the proper input beam parameters command."""
        _tracewin_command = self._create_tracewin_command()
        return _tracewin_command

    def _create_tracewin_command(self, warn_missing_phase_space: bool = True
                                 ) -> list[str]:
        """
        Turn emittance, alpha, beta from the proper phase-spaces into command.

        When phase-spaces were not created, we return np.NaN which will
        ultimately lead TraceWin to take this data from its `.ini` file.

        """
        args = []
        for phase_space_name in ['x', 'y', 'z']:
            if phase_space_name not in self.__dir__():
                eps, alpha, beta = np.NaN, np.NaN, np.NaN

                phase_spaces_are_needed = \
                    (isinstance(self.z_abs, np.ndarray)
                        and self.z_abs[0] > 1e-10) \
                    or (isinstance(self.z_abs, float) and self.z_abs > 1e-10)

                if warn_missing_phase_space and phase_spaces_are_needed:
                    logging.warning(f"{phase_space_name} phase space not "
                                    "defined, keeping default inputs from the "
                                    "`.ini.`.")
            else:
                phase_space = getattr(self, phase_space_name)
                eps, alpha, beta = _to_float_if_necessary(
                    *phase_space.get('eps', 'alpha', 'beta')
                )

            args.extend((eps, alpha, beta))
        return beam_parameters_to_command(*args)

    def create_phase_spaces(self,
                            *args: str,
                            **kwargs: dict[str, np.ndarray | float | None]
                            ) -> None:
        """
        Recursively create the phase spaces with their initial values.

        Parameters
        ----------
        *args : str
            Name of the phase spaces to be created.
        **kwargs : dict[str, np.ndarray | float]
            Keyword arguments to directly initialize properties in some phase
            spaces. Name of the keyword argument must correspond to a phase
            space. Argument must be a dictionary, which keys must be
            understandable by SinglePhaseSpaceBeamParameters.__init__: alpha,
            beta, gamma, eps, twiss, envelope_pos and envelope_energy are
            allowed values.

        """
        for arg in args:
            phase_space_kwargs = kwargs.get(arg, None)
            if phase_space_kwargs is None:
                phase_space_kwargs = {}

            phase_space_beam_param = SinglePhaseSpaceBeamParameters(
                arg,
                element_to_index=self.element_to_index,
                **phase_space_kwargs,
            )
            setattr(self, arg, phase_space_beam_param)

    def init_other_phase_spaces_from_zdelta(
            self, *args: str, gamma_kin: np.ndarray | None = None,
            beta_kin: np.ndarray | None = None) -> None:
        """Create the desired longitudinal planes from zdelta."""
        if gamma_kin is None:
            gamma_kin = self.gamma_kin
        if beta_kin is None:
            beta_kin = self.beta_kin
        args_for_init = (self.zdelta.eps, self.zdelta.twiss, gamma_kin,
                         beta_kin)

        for arg in args:
            if arg not in ('phiw', 'z'):
                logging.error(f"Phase space conversion zdelta -> {arg} not "
                              "implemented. Ignoring...")

        if 'phiw' in args:
            self.phiw.init_from_another_plane(*args_for_init, 'zdelta to phiw')
        if 'z' in args:
            self.z.init_from_another_plane(*args_for_init, 'zdelta to z')

    def init_99percent_phase_spaces(self, eps_phiw99: np.ndarray,
                                    eps_x99: np.ndarray, eps_y99: np.ndarray
                                    ) -> None:
        """Set 99% emittances; envelopes, Twiss, etc not implemented."""
        self.phiw99.eps = eps_phiw99
        self.x99.eps = eps_x99
        self.y99.eps = eps_y99


@dataclass
class SinglePhaseSpaceBeamParameters:
    """Hold Twiss, emittance, envelopes of a single phase-space."""

    phase_space: str

    sigma_in: np.ndarray | None = None
    sigma: np.ndarray | None = None
    tm_cumul: np.ndarray | None = None

    twiss: np.ndarray | None = None

    alpha: np.ndarray | float | None = None
    beta: np.ndarray | float | None = None
    gamma: np.ndarray | None = None

    eps: np.ndarray | float | None = None
    envelope_pos: np.ndarray | None = None
    envelope_energy: np.ndarray | None = None

    mismatch_factor: np.ndarray | None = None

    element_to_index: Callable[[str | Element, str | None], int | slice] \
        | None = None

    def __post_init__(self):
        """Set the default attributes for the zdelta."""
        if self.phase_space == 'zdelta' and self.sigma_in is None:
            # logging.warning("resorted back to a default sigma_zdelta. I should"
            #                 "avoid that.")
            self.sigma_in = con.SIGMA_ZDELTA

    def __str__(self) -> str:
        """Show amplitude of some of the attributes."""
        out = f"\tSinglePhaseSpaceBeamParameters {self.phase_space}:\n"
        for key in ('alpha', 'beta', 'eps', 'envelope_pos', 'envelope_energy',
                    'mismatch_factor'):
            out += "\t\t" + range_vals_object(self, key)
        return out

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: Element | None = None, pos: str | None = None,
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

            val[key] = recursive_getter(key, vars(self), to_numpy=False,
                                        none_to_nan=False, **kwargs)

            if val[key] is None:
                continue

            if None not in (self.element_to_index, elt):
                idx = self.element_to_index(elt=elt, pos=pos)
                val[key] = val[key][idx]

        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) if isinstance(val, list) else val
                   for val in out]
            if none_to_nan:
                out = [val.astype(float) for val in out]

        if len(out) == 1:
            return out[0]
        return tuple(out)

    def init_from_cumulated_transfer_matrices(
            self, gamma_kin: np.ndarray, tm_cumul: np.ndarray | None = None,
            beta_kin: np.ndarray | None = None) -> None:
        """
        Use transfer matrices to compute `sigma`, and then everything.

        Used by the Envelope1D solver.
        """
        if self.tm_cumul is None and tm_cumul is None:
            logging.error("Missing `tm_cumul` to compute beam parameters.")
            return

        if self.tm_cumul is None:
            self.tm_cumul = tm_cumul
        if tm_cumul is None:
            tm_cumul = self.tm_cumul
        if beta_kin is None:
            beta_kin = converters.energy(gamma_kin, 'gamma to beta')

        self.sigma = _sigma_beam_matrices(tm_cumul, self.sigma_in)
        self.init_from_sigma(gamma_kin, beta_kin)

    def init_from_sigma(self, gamma_kin: np.ndarray, beta_kin: np.ndarray,
                        sigma: np.ndarray | None = None) -> None:
        """
        Compute Twiss, eps, envelopes just from sigma matrix.

        Used by the Envelope1D and TraceWin solvers.

        """
        if sigma is None:
            sigma = self.sigma

        eps_no_normalisation, eps_normalized = self._compute_eps_from_sigma(
            sigma, gamma_kin, beta_kin)
        self.eps = eps_normalized
        self._eps_no_norm = eps_no_normalisation

        self._compute_twiss_from_sigma(sigma, eps_no_normalisation)
        self.envelope_pos, self.envelope_energy = \
            self._compute_envelopes_from_sigma(sigma)

    def reconstruct_full_sigma_matrix(
        self, sigma_00: np.ndarray, sigma_01: np.ndarray, eps: np.ndarray,
        eps_is_normalized: bool = True, gamma_kin: np.ndarray | None = None,
        beta_kin: np.ndarray | None = None,
    ) -> None:
        """
        Compute sigma matrix from the two top components and emittance.

        sigma matrix is always saved in SI units (m, rad).

        Used by the TraceWin solver.
        For the zdelta phase space:
            Inputs must be in "practical" units: mm, mrad.
            ! Note that epsilon_zdelta is in pi.mm.mrad in TraceWin .out files,
            ! while pi.mm.% is used everywhere else in TraceWin as well as in
            ! LightWin.
        For the transverse phase spaces, units are mm and mrad.

        Parameters
        ----------
        sigma_00 : np.ndarray
            Top-left component of the sigma matrix.
        sigma_01 : np.ndarray
            Top-right = bottom-left component of the sigma matrix.
        eps : np.ndarray
            Emittance.
        eps_is_normalized : bool, optional
            To tell if the given emittance is already normalized. The default
            is True. In this case, it is de-normalized and `gamma_kin` must be
            given.
        gamma_kin : np.ndarray | None, optional
            Lorentz gamma factor. The default is None. It is however mandatory
            if the emittance is given unnormalized.
        beta_kin : np.ndarray | None, optional
            Lorentz beta factor. The default is None. In this case, we compute
            it from gamma_kin.

        """
        if self.phase_space not in ['zdelta', 'x', 'y', 'x99', 'y99']:
            logging.warning("sigma reconstruction in this phase space not "
                            "tested. The main issue that can appear is with "
                            "units.")

        sigma = _reconstruct_full_sigma_matrix(sigma_00, sigma_01, eps,
                                               eps_is_normalized=True,
                                               gamma_kin=gamma_kin,
                                               beta_kin=beta_kin)
        if self.phase_space in ['zdelta', 'x', 'y', 'x99', 'y99']:
            sigma *= 1e-6
        self.sigma = sigma

    def init_from_another_plane(self, eps_orig: np.ndarray,
                                twiss_orig: np.ndarray, gamma_kin: np.ndarray,
                                beta_kin: np.ndarray, convert: str) -> None:
        """
        Fully initialize from another phase space.

        Used by the Envelope1D and TraceWin solvers.

        """
        eps_no_normalisation, eps_normalized = \
            self._compute_eps_from_other_plane(eps_orig, convert, gamma_kin,
                                               beta_kin)
        self.eps = eps_normalized
        self._compute_twiss_from_other_plane(twiss_orig, convert, gamma_kin,
                                             beta_kin)
        eps_for_envelope = eps_no_normalisation
        if self.phase_space == 'phiw':
            eps_for_envelope = eps_normalized
        self.compute_envelopes(self.twiss[:, 1], self.twiss[:, 2],
                               eps_for_envelope)

    def init_from_averaging_x_and_y(self, x_space: object, y_space: object
                                    ) -> None:
        """Create eps for an average transverse plane phase space."""
        self.eps = .5 * (x_space.eps + y_space.eps)
        if None not in (x_space.mismatch_factor, y_space.mismatch_factor):
            self.mismatch_factor = .5 * (x_space.mismatch_factor
                                         + y_space.mismatch_factor)
        self.twiss = None
        self.envelope_pos, self.envelope_energy = None, None

    def _compute_eps_from_sigma(self, sigma: np.ndarray, gamma_kin: np.ndarray,
                                beta_kin: np.ndarray) -> tuple[np.ndarray,
                                                               np.ndarray]:
        """
        Compute emittance from sigma matrix.

        For the zdelta phase space:
            sigma is in SI units
            emittance is returned in pi.mm.%
        For the transverse phase spaces:
            sigma is in SI units
            emittances is returned in pi.mm.mrad

        Parameters
        ----------
        sigma : np.ndarray
            Sigma matrix in SI units.
        gamma_kin : np.ndarray
            Lorentz gamma factor.
        beta_kin : np.ndarray
            Lorentz beta factor.

        Returns
        -------
        eps_no_normalisation : np.ndarray
            Emittance not normalized.
        eps_normalized : np.ndarray
            Emittance normalized.

        """
        assert self.phase_space in ['zdelta', 'x', 'y', 'x99', 'y99']
        eps_no_normalisation = np.array(
            [np.sqrt(np.linalg.det(sigma[i])) for i in range(sigma.shape[0])])

        if self.phase_space in ['zdelta']:
            eps_no_normalisation *= 1e5
        elif self.phase_space in ['x', 'y', 'x99', 'y99']:
            eps_no_normalisation *= 1e6

        eps_normalized = converters.emittance(eps_no_normalisation,
                                              f"normalize {self.phase_space}",
                                              gamma_kin=gamma_kin,
                                              beta_kin=beta_kin)
        return eps_no_normalisation, eps_normalized

    def _compute_eps_from_other_plane(self, eps_orig: np.ndarray, convert: str,
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

    def _compute_twiss_from_sigma(self, sigma: np.ndarray,
                                  eps_no_normalisation: np.ndarray
                                  ) -> None:
        """
        Compute the Twiss parameters using the sigma matrix.

        For the zdelta phase space:
            sigma is in SI units
            eps_no_normalisation should be in pi.mm.%
        For the transverse planes:
            sigma is in SI units
            eps_no_normalisation is in pi.mm.mrad

        Parameters
        ----------
        sigma : np.ndarray
            sigma matrix along the linac.
        eps_no_normalisation : np.ndarray
            Unnormalized emittance.

        Returns
        -------
        Nothing, but set attributes alpha (no units), beta (mm/pi.%), gamma
        (mm.pi/%) and twiss (column_stacking of the three), in the z-dp/p
        plane.
        In the transverse planes, units are pi mm and mrad instead.

        """
        assert self.phase_space in ['zdelta', 'x', 'y', 'x99', 'y99']
        assert self.eps is not None
        n_points = sigma.shape[0]
        twiss = np.full((n_points, 3), np.NaN)

        for i in range(n_points):
            twiss[i, :] = np.array(
                [-sigma[i][1, 0], sigma[i][0, 0], sigma[i][1, 1]]
            ) / eps_no_normalisation[i] * 1e6

        if self.phase_space == 'zdelta':
            twiss[:, 0] *= 1e-1
            twiss[:, 2] *= 1e-2

        self._unpack_twiss(twiss)
        self.twiss = twiss

    def _compute_twiss_from_other_plane(self, twiss_orig: np.ndarray,
                                        convert: str, gamma_kin: np.ndarray,
                                        beta_kin: np.ndarray) -> None:
        """Compute Twiss parameters from Twiss parameters in another plane."""
        self.twiss = converters.twiss(twiss_orig, gamma_kin, convert,
                                      beta_kin=beta_kin)
        self._unpack_twiss(self.twiss)

    # TODO would be possible to skip this with TW, where envelope_pos is
    # already known
    def _compute_envelopes_from_sigma(self, sigma: np.ndarray
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

    def _unpack_twiss(self, twiss: np.ndarray) -> None:
        """Unpack a three-columns twiss array in alpha, beta, gamma."""
        self.alpha = twiss[:, 0]
        self.beta = twiss[:, 1]
        self.gamma = twiss[:, 2]


# =============================================================================
# Public
# =============================================================================
def mismatch_from_objects(ref: BeamParameters, fix: BeamParameters,
                          *phase_spaces: str,
                          set_transverse_as_average: bool = True) -> None:
    """Compute the mismatchs in the desired phase_spaces."""
    z_ref, z_fix = ref.z_abs, fix.z_abs
    for phase_space in phase_spaces:
        bp_ref, bp_fix = getattr(ref, phase_space), getattr(fix, phase_space)
        bp_fix.mismatch_factor = _mismatch_single_phase_space(bp_ref, bp_fix,
                                                              z_ref, z_fix)

    if not set_transverse_as_average:
        return

    if 'x' not in phase_spaces or 'y' not in phase_spaces:
        logging.warning("Transverse planes were not updated. Transverse "
                        "mismatch may be meaningless.")

    fix.t.mismatch_factor = .5 * (fix.x.mismatch_factor
                                  + fix.y.mismatch_factor)


def _mismatch_single_phase_space(ref: SinglePhaseSpaceBeamParameters,
                                 fix: SinglePhaseSpaceBeamParameters,
                                 z_ref: np.ndarray,
                                 z_fix: np.ndarray
                                 ) -> np.ndarray | None:
    """Compute the mismatch using two `SinglePhaseSpaceBeamParameters`."""
    twiss_ref, twiss_fix = ref.twiss, fix.twiss
    if twiss_ref.shape != twiss_fix.shape:
        twiss_ref = _resample_twiss_on_fix(z_ref, twiss_ref, z_fix)

    mism = mismatch_from_arrays(twiss_ref, twiss_fix, transp=True)
    return mism


def mismatch_from_arrays(ref: np.ndarray, fix: np.ndarray, transp: bool = False
                         ) -> np.ndarray:
    """Compute the mismatch factor between two ellipses."""
    assert isinstance(ref, np.ndarray)
    assert isinstance(fix, np.ndarray)
    # Transposition needed when computing the mismatch factor at more than one
    # position, as shape of twiss arrays is (n, 3).
    if transp:
        ref = ref.transpose()
        fix = fix.transpose()

    # R in TW doc
    __r = ref[1] * fix[2] + ref[2] * fix[1]
    __r -= 2. * ref[0] * fix[0]

    # Forbid R values lower than 2 (numerical error)
    __r = np.atleast_1d(__r)
    __r[np.where(__r < 2.)] = 2.

    mismatch = np.sqrt(.5 * (__r + np.sqrt(__r**2 - 4.))) - 1.
    return mismatch


def _reconstruct_full_sigma_matrix(
    sigma_00: np.ndarray, sigma_01: np.ndarray, eps: np.ndarray,
    eps_is_normalized: bool = True, gamma_kin: np.ndarray | None = None,
    beta_kin: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute sigma matrix from the two top components and emittance.

    For consistency with TraceWin results files, inputs must be in the
    z-delta phase space, in "practical" units: mm, mrad.
      | Note that epsilon_zdelta is in pi.mm.mrad in TraceWin .out files,
      | while pi.mm.% is used everywhere else in TraceWin as well as in
      | LightWin.

    For consistency with Envelope1D, output is in z-delta phase space, in
    SI units: m, rad.

    Parameters
    ----------
    sigma_00 : np.ndarray
        Top-left component of the sigma matrix.
    sigma_01 : np.ndarray
        Top-right = bottom-left component of the sigma matrix.
    eps : np.ndarray
        Emittance.
    eps_is_normalized : bool, optional
        To tell if the given emittance is already normalized. The default is
        True. In this case, it is de-normalized and `gamma_kin` must be given.
    gamma_kin : np.ndarray | None, optional
        Lorentz gamma factor. The default is None. It is however mandatory if
        the emittance is given unnormalized.
    beta_kin : np.ndarray | None, optional
        Lorentz beta factor. The default is None. In this case, we compute it
        from gamma_kin.

    Returns
    -------
    sigma : np.ndarray
        Full sigma matrix along the linac.

    """
    if eps_is_normalized:
        if gamma_kin is None:
            logging.error("It is mandatory to give `gamma_kin` to compute "
                          "sigma matrix. Aborting calculation of this phase "
                          "space...")
            return

        if beta_kin is None:
            beta_kin = converters.energy(gamma_kin, 'gamma to beta')
        eps /= (beta_kin * gamma_kin)

    sigma = np.zeros((sigma_00.shape[0], 2, 2))
    sigma[:, 0, 0] = sigma_00
    sigma[:, 0, 1] = sigma_01
    sigma[:, 1, 0] = sigma_01
    sigma[:, 1, 1] = (eps**2 + sigma_01**2) / sigma_00
    return sigma  # zdelta* 1e-6


# =============================================================================
# Private
# =============================================================================
def _sigma_beam_matrices(tm_cumul: np.ndarray, sigma_in: np.ndarray
                         ) -> np.ndarray:
    """
    Compute the sigma beam matrices between over the linac.

    sigma_in and transfer matrices should be in the same ref. By default,
    LW calculates transfer matrices in [z - delta].
    """
    sigma = []
    n_points = tm_cumul.shape[0]

    for i in range(n_points):
        sigma.append(tm_cumul[i] @ sigma_in @ tm_cumul[i].transpose())
    return np.array(sigma)


def _phase_space_name_hidden_in_key(key: str) -> bool:
    """Look for the name of a phase-space in a key name."""
    if '_' not in key:
        return False

    to_test = key.split('_')
    if to_test[-1] in PHASE_SPACES:
        return True
    return False


def _separate_var_from_phase_space(key: str) -> tuple[str, str]:
    """Separate variable name from phase space name."""
    splitted = key.split('_')
    key = '_'.join(splitted[:-1])
    phase_space = splitted[-1]
    return key, phase_space


def _to_float_if_necessary(eps: float | np.ndarray, alpha: float | np.ndarray,
                           beta: float | np.ndarray
                           ) -> tuple[float, float, float]:
    """Ensure that the data given to TraceWin will be float."""
    are_floats = [isinstance(parameter, float)
                  for parameter in (eps, alpha, beta)]
    if all(are_floats):
        return eps, alpha, beta

    logging.warning("You are trying to give TraceWin an array of eps, alpha or"
                    " beta, while it should be a float. I suspect that the "
                    "current `BeamParameters` was generated by a "
                    "`SimulationOutuput`, while it should have been created "
                    "by a `ListOfElements` (initial beam state). Taking "
                    "first element of each array...")

    output_as_floats = [parameter if is_float else parameter[0]
                        for parameter, is_float in zip((eps, alpha, beta),
                                                       are_floats)]
    return tuple(output_as_floats)


def _resample_twiss_on_fix(z_ref: np.ndarray, twiss_ref: np.ndarray,
                           z_fix: np.ndarray) -> np.ndarray:
    """Interpolate ref Twiss on fix Twiss to compute mismatch afterwards."""
    n_points = z_fix.shape[0]
    out = np.empty((n_points, 3))

    for axis in range(out.shape[1]):
        out[:, axis] = np.interp(z_fix, z_ref, twiss_ref[:, axis])
    return out
