#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather beam parameters at the entrance of a :class:`.ListOfElements`.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from typing import Any
from dataclasses import dataclass
import logging

import numpy as np

import config_manager as con

from core.beam_parameters.phase_space_initial_beam_parameters import (
    PhaseSpaceInitialBeamParameters,
    IMPLEMENTED_PHASE_SPACES,
)
from core.elements.element import Element

from tracewin_utils.interface import beam_parameters_to_command

from util.helper import recursive_items
from util.dicts_output import markdown


@dataclass
class InitialBeamParameters:
    r"""
    Hold all emittances, envelopes, etc in various planes.

    Attributes
    ----------
    z_abs : float
        Absolute position in the linac in m.
    gamma_kin : float
        Lorentz gamma factor.
    beta_kin : float
        Lorentz gamma factor.
    zdelta, z, phiw, x, y, t : PhaseSpaceInitialBeamParameters
        Holds beam parameters respectively in the :math:`[z-z\delta]`,
        :math:`[z-z']`, :math:`[\phi-W]`, :math:`[x-x']`, :math:`[y-y']` and
        :math:`[t-t']` planes.
    phiw99, x99, y99 : PhaseSpaceInitialBeamParameters
        Holds 99% beam parameters respectively in the :math:`[\phi-W]`,
        :math:`[x-x']` and :math:`[y-y']` planes. Only used with multiparticle
        simulations.
    sigma_in : np.ndarray | None, optional
        Holds the (6, 6) in 1D simulation) :math:`\sigma` beam matrix at the
        entrance of the linac/portion of linac. The default is None.

    """

    phase_spaces: tuple[str, ...]
    z_abs: float
    gamma_kin: float
    beta_kin: float
    sigma_in: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Define the attributes that may be used."""
        self.zdelta: PhaseSpaceInitialBeamParameters
        self.z: PhaseSpaceInitialBeamParameters
        self.phiw: PhaseSpaceInitialBeamParameters
        self.x: PhaseSpaceInitialBeamParameters
        self.y: PhaseSpaceInitialBeamParameters
        self.t: PhaseSpaceInitialBeamParameters
        self.phiw99: PhaseSpaceInitialBeamParameters
        self.x99: PhaseSpaceInitialBeamParameters
        self.y99: PhaseSpaceInitialBeamParameters

        self.create_phase_spaces(*self.phase_spaces)

    def init_phase_spaces_from_sigma(self, sub_sigmas: dict[str, np.ndarray]
                                     ) -> None:
        r"""Init phase space from a sigma matrix.

        Parameters
        ----------
        sub_sigmas : dict[str, np.ndarray]
            Dict where keys are name of phase space, values are :math:`\sigma`
            beam matrix in corresponding plane. Shape is (2, 2).

        """
        for phase_space_name in self.phase_spaces:
            phase_space = self.get(phase_space_name)
            sub_sigma = sub_sigmas[phase_space_name]
            phase_space.init_from_sigma(sub_sigma,
                                        self.gamma_kin,
                                        self.beta_kin)
            phase_space.init_eye_tm_cumul()

    def init_phase_spaces_from_kwargs(self, initial_beam_kwargs) -> None:
        """Init phase spaces from a dictionary.

        Parameters
        ----------
        initial_beam_kwargs : dict[str, dict[str, float | np.ndarray]]
            Keys should be the name of a phase space.
            The values should be other dictionaries, which keys-values are
            :class:`.PhaseSpaceInitialBeamParameters` attributes.

        """
        for key, value in initial_beam_kwargs.items():
            assert key in IMPLEMENTED_PHASE_SPACES, f"{key = } should be the "\
                "name of a phase space."
            setattr(self, key, PhaseSpaceInitialBeamParameters(key, **value))

    def create_phase_spaces(self,
                            *args: str,
                            **kwargs: np.ndarray | float | None
                            ) -> None:
        """
        Recursively create the phase spaces with their initial values.

        Parameters
        ----------
        *args : str
            Name of the phase spaces to be created.
        **kwargs : np.ndarray | float | None
            Keyword arguments to directly initialize properties in some phase
            spaces. Name of the keyword argument must correspond to a phase
            space. Argument must be a dictionary, which keys must be
            understandable by :meth:`PhaseSpaceInitialBeamParameters.__init__`:
            ``'alpha'``, ``'beta'``, ``'gamma'``, ``'eps'``, ``'twiss'``,
            ``'envelope_pos'`` and ``'envelope_energy'`` are allowed values.

        """
        logging.warning(f"Creating phase spaces {args}")

        sigma_in = self._format_sigma_in()
        phase_space_to_proper_sigma_in = {
            'x': sigma_in[:2, :2],
            'y': sigma_in[2:4, 2:4],
            'zdelta': sigma_in[4:, 4:],
        }

        for phase_space_name in args:
            kwargs_for_this_phase_space = kwargs.get(phase_space_name, None)
            if kwargs_for_this_phase_space is None:
                kwargs_for_this_phase_space = {}
            proper_sigma_in = phase_space_to_proper_sigma_in.get(
                phase_space_name,
                None)
            phase_space_beam_param = PhaseSpaceInitialBeamParameters(
                phase_space_name,
                sigma_in=proper_sigma_in,
                **kwargs_for_this_phase_space,
            )
            setattr(self, phase_space_name, phase_space_beam_param)

    def __str__(self) -> str:
        """Give compact information on the data that is stored."""
        out = "\tBeamParameters:\n"
        out += "\t\t" + markdown["eps_zdelta"] + str(self.zdelta.eps)
        out += "\t\t" + markdown["beta_zdelta"] + str(self.zdelta.beta)
        out += "\t\t" + markdown["mismatch_zdelta"] + \
            str(self.zdelta.mismatch_factor)
        return out

    def has(self, key: str) -> bool:
        """
        Tell if the required attribute is in this class.

        Notes
        -----
        ``key = 'property_phasespace'`` will return True if ``'property'``
        exists in ``phasespace``. Hence, the following two commands will have
        the same return values:

            .. code-block:: python

                self.has('twiss_zdelta')
                self.zdelta.has('twiss')

        See Also
        --------
        get

        """
        if _phase_space_name_hidden_in_key(key):
            key, phase_space_name = _separate_var_from_phase_space(key)
            phase_space = getattr(self, phase_space_name)
            return hasattr(phase_space, key)
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: Element | None = None, pos: str | None = None,
            phase_space: str | None = None, **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Notes
        -----
        What is particular in this getter is that all
        :class:`PhaseSpaceInitialBeamParameters` attributes have attributes with
        the same name: ``twiss``, ``alpha``, ``beta``, ``gamma``, ``eps``,
        ``envelopes_pos`` and ``envelopes_energy``.

        Hence, you must provide either a ``phase_space`` argument which shall
        be in :data:`IMPLEMENTED_PHASE_SPACES`, either you must append the
        name of the phase space to the name of the desired variable with an
        underscore.

        See Also
        --------
        has

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
            Otherwise, it must be in :data:`IMPLEMENTED_PHASE_SPACES`.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val = {key: [] for key in keys}
        # Explicitely look into a specific PhaseSpaceInitialBeamParameters
        if phase_space is not None:
            single_phase_space_beam_param = getattr(self, phase_space)
            return single_phase_space_beam_param.get(*keys,
                                                     elt=elt,
                                                     pos=pos,
                                                     **kwargs)
        for key in keys:
            # Look for key in PhaseSpaceInitialBeamParameters
            if _phase_space_name_hidden_in_key(key):
                short_key, phase_space = _separate_var_from_phase_space(key)
                assert hasattr(self, phase_space), f"{phase_space = } not set"\
                    + " for current BeamParameters object."
                single_phase_space_beam_param = getattr(self, phase_space)
                val[key] = single_phase_space_beam_param.get(short_key)
                continue

            # Look for key in BeamParameters
            if not self.has(key):
                val[key] = None
                continue
            val[key] = getattr(self, key)

        if elt is not None:
            assert self.element_to_index is not None
            idx = self.element_to_index(elt=elt, pos=pos)
            val = {_key: _value[idx] for _key, _value in val.items()}

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
        ultimately lead TraceWin to take this data from its ``.ini`` file.

        """
        args = []
        for phase_space_name in ('x', 'y', 'z'):
            if not self.has(phase_space_name):
                eps, alpha, beta = np.NaN, np.NaN, np.NaN
                phase_spaces_are_needed = self.z_abs > 1e-10
                if warn_missing_phase_space and phase_spaces_are_needed:
                    logging.warning(f"{phase_space_name} phase space not "
                                    "defined, keeping default inputs from the "
                                    "`.ini.`.")
            else:
                phase_space = getattr(self, phase_space_name)
                eps, alpha, beta = phase_space.get('eps', 'alpha', 'beta')

            args.extend((eps, alpha, beta))
        return beam_parameters_to_command(*args)

    def _format_sigma_in(self) -> np.ndarray:
        r"""Format the input :math:`\sigma` beam matrix for uniformity.

        Returns
        -------
        sigma_in : np.ndarray
            (6, 6) sigma beam matrix filled with np.NaN where data is missing.

        .. deprecated:: v3.2.2.3
            This matrix should always be set, and always have a (6, 6) shape.
            This method will be removed in the future.

        """
        if self.sigma_in is None:
            if hasattr(con, 'SIGMA'):
                sigma_in_from_conf = con.SIGMA
                if ~np.isnan(sigma_in_from_conf).any():
                    logging.warning("Initialized sigma beam matrix from config"
                                    " manager. Please give it to "
                                    "BeamParameters.__init__ instead."
                                    "Ignore this if solver is TW.")
                    return sigma_in_from_conf

            logging.warning("Initialized sigma beam matrix from config"
                            " manager. Please give it to "
                            "BeamParameters.__init__ instead."
                            "Also, should use SIGMA instead of "
                            "SIGMA_ZDELTA which will be deprecated.")
            sigma_in_from_conf = con.SIGMA_ZDELTA
            sigma_in = np.full((6, 6), np.NaN)
            sigma_in[4:, 4:] = sigma_in_from_conf
            return sigma_in

        shape = self.sigma_in.shape
        if shape == (6, 6):
            return self.sigma_in

        if shape == (2, 2):
            sigma_in = np.full((6, 6), np.NaN)
            sigma_in[4:, 4:] = self.sigma_in
            return sigma_in

        raise IOError("Given sigma_in was not understood.")


# =============================================================================
# Private
# =============================================================================
def _phase_space_name_hidden_in_key(key: str) -> bool:
    """Look for the name of a phase-space in a key name."""
    if '_' not in key:
        return False

    to_test = key.split('_')
    if to_test[-1] in IMPLEMENTED_PHASE_SPACES:
        return True
    return False


def _separate_var_from_phase_space(key: str) -> tuple[str, str]:
    """Separate variable name from phase space name."""
    splitted = key.split('_')
    key = '_'.join(splitted[:-1])
    phase_space = splitted[-1]
    return key, phase_space
