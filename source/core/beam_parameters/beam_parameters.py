#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather the beam parameters of all the phase spaces.

For a list of the units associated with every parameter, see
:ref:`units-label`.

.. todo::
    Recheck the _create_tracewin_command. ``phases_are_needed`` = not clear.
    Why use z and not zdelta? Needed here... Or rather in
    initial_beam_parameters?

.. todo::
    Initialization of phase spaces directly in the __init__ method.

"""
from typing import Any, Callable
from dataclasses import dataclass
import logging

import numpy as np

import config_manager as con

from core.beam_parameters.phase_space_beam_parameters import (
    PhaseSpaceBeamParameters,
    mismatch_single_phase_space,
    IMPLEMENTED_PHASE_SPACES,
)
from core.elements.element import Element

from tracewin_utils.interface import beam_parameters_to_command

from util.helper import (recursive_items,
                         range_vals,
                         )


@dataclass
class BeamParameters:
    r"""
    Hold all emittances, envelopes, etc in various planes.

    Attributes
    ----------
    z_abs : np.ndarray
        Absolute position in the linac in m.
    gamma_kin : np.ndarray
        Lorentz gamma factor.
    beta_kin : np.ndarray
        Lorentz gamma factor.
    element_to_index : Callable[[str | Element, str | None],
                                 int | slice] | None, optional
        Takes an :class:`.Element`, its name, ``'first'`` or ``'last'`` as
        argument, and returns corresponding index. Index should be the same in
        all the arrays attributes of this class: ``z_abs``, ``beam_parameters``
        attributes, etc. Used to easily ``get`` the desired properties at the
        proper position. The default is None.
    zdelta, z, phiw, x, y, t : PhaseSpaceBeamParameters
        Holds beam parameters respectively in the :math:`[z-z\delta]`,
        :math:`[z-z']`, :math:`[\phi-W]`, :math:`[x-x']`, :math:`[y-y']` and
        :math:`[t-t']` planes.
    phiw99, x99, y99 : PhaseSpaceBeamParameters
        Holds 99% beam parameters respectively in the :math:`[\phi-W]`,
        :math:`[x-x']` and :math:`[y-y']` planes. Only used with multiparticle
        simulations.
    sigma_in : np.ndarray | None, optional
        Holds the (6, 6) :math:`\sigma` beam matrix at the entrance of the
        linac/portion of linac.  The default is None.
    n_points : int | None, optional
        Holds the number of points along the linac (1 if the object defines a
        beam at the entry of the linac/a linac portion). The default is None.

    """

    z_abs: np.ndarray
    gamma_kin: np.ndarray
    beta_kin: np.ndarray
    element_to_index: Callable[[str | Element, str | None], int | slice] \
        | None = None
    sigma_in: np.ndarray | None = None
    n_points: int | None = None

    def __post_init__(self) -> None:
        """Define the attributes that may be used."""
        if self.n_points is not None:
            logging.warning("giving n_points to BeamParameters.__init__ is "
                            "deprecated")
        self.n_points = np.atleast_1d(self.z_abs).shape[0]
        self.zdelta: PhaseSpaceBeamParameters
        self.z: PhaseSpaceBeamParameters
        self.phiw: PhaseSpaceBeamParameters
        self.x: PhaseSpaceBeamParameters
        self.y: PhaseSpaceBeamParameters
        self.t: PhaseSpaceBeamParameters
        self.phiw99: PhaseSpaceBeamParameters
        self.x99: PhaseSpaceBeamParameters
        self.y99: PhaseSpaceBeamParameters

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

    def get(self,
            *keys: str,
            to_numpy: bool = True,
            none_to_nan: bool = False,
            elt: Element | None = None,
            pos: str | None = None,
            phase_space: str | None = None,
            **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Notes
        -----
        What is particular in this getter is that all
        :class:`PhaseSpaceBeamParameters` attributes have attributes with
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
        # Explicitely look into a specific PhaseSpaceBeamParameters
        if phase_space is not None:
            single_phase_space_beam_param = getattr(self, phase_space)
            return single_phase_space_beam_param.get(*keys,
                                                     elt=elt,
                                                     pos=pos,
                                                     **kwargs)
        for key in keys:
            # Look for key in PhaseSpaceBeamParameters
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

    @property
    def sigma(self) -> np.ndarray:
        """Give value of sigma.

        .. todo::
            Could be cleaner.

        """
        assert isinstance(self.n_points, int)
        sigma = np.zeros((self.n_points, 6, 6))

        sigma_x = np.zeros((self.n_points, 2, 2))
        if self.has('x') and self.x.is_set('sigma'):
            sigma_x = self.x.sigma

        sigma_y = np.zeros((self.n_points, 2, 2))
        if self.has('y') and self.y.is_set('sigma'):
            sigma_y = self.y.sigma

        sigma_zdelta = self.zdelta.sigma

        sigma[:, :2, :2] = sigma_x
        sigma[:, 2:4, 2:4] = sigma_y
        sigma[:, 4:, 4:] = sigma_zdelta
        return sigma

    @sigma.setter
    def sigma(self, value: np.ndarray) -> None:
        logging.warning("You shall not set sigma directly, but rather the "
                        "sub-sigma matrices in x, y, zdelta planes.")
        assert value.shape == (self.n_points, 6, 6)
        self.x.sigma = value[:, :2, :2]
        self.y.sigma = value[:, 2:4, 2:4]
        self.zdelta.sigma = value[:, 4:, 4:]

    def _create_tracewin_command(self, warn_missing_phase_space: bool = True
                                 ) -> list[str]:
        """
        Turn emittance, alpha, beta from the proper phase-spaces into command.

        When phase-spaces were not created, we return np.NaN which will
        ultimately lead TraceWin to take this data from its ``.ini`` file.

        """
        args = []
        for phase_space_name in ('x', 'y', 'z'):
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
            understandable by :meth:`PhaseSpaceBeamParameters.__init__`:
            ``'alpha'``, ``'beta'``, ``'gamma'``, ``'eps'``, ``'twiss'``,
            ``'envelope_pos'`` and ``'envelope_energy'`` are allowed values.

        """
        sigma_in = self._format_sigma_in()
        phase_space_to_proper_sigma_in = {
            'x': sigma_in[:2, :2],
            'y': sigma_in[2:4, 2:4],
            'zdelta': sigma_in[4:, 4:],
        }

        for phase_space_name in args:
            this_phase_space_kw = kwargs.get(phase_space_name, {})
            proper_sigma_in = phase_space_to_proper_sigma_in.get(
                phase_space_name,
                None)
            phase_space_beam_param = PhaseSpaceBeamParameters(
                phase_space_name,
                element_to_index=self.element_to_index,
                sigma_in=proper_sigma_in,
                n_points=self.n_points,
                **this_phase_space_kw,
            )
            setattr(self, phase_space_name, phase_space_beam_param)

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
# Public
# =============================================================================
def mismatch_from_objects(ref: BeamParameters,
                          fix: BeamParameters,
                          *phase_spaces: str,
                          set_transverse_as_average: bool = True) -> None:
    """Compute the mismatchs in the desired phase_spaces."""
    z_ref, z_fix = ref.z_abs, fix.z_abs
    for phase_space in phase_spaces:
        bp_ref, bp_fix = getattr(ref, phase_space), getattr(fix, phase_space)
        bp_fix.mismatch_factor = mismatch_single_phase_space(bp_ref, bp_fix,
                                                             z_ref, z_fix)

    if not set_transverse_as_average:
        return

    if 'x' not in phase_spaces or 'y' not in phase_spaces:
        logging.warning("Transverse planes were not updated. Transverse "
                        "mismatch may be meaningless.")

    fix.t.mismatch_factor = .5 * (fix.x.mismatch_factor
                                  + fix.y.mismatch_factor)


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


def _to_float_if_necessary(eps: float | np.ndarray,
                           alpha: float | np.ndarray,
                           beta: float | np.ndarray
                           ) -> tuple[float, float, float]:
    """
    Ensure that the data given to TraceWin will be float.

        .. deprecated:: v3.2.2.3
            eps, alpha, beta will always be arrays of size 1.

    """
    as_arrays = (np.atleast_1d(eps), np.atleast_1d(alpha), np.atleast_1d(beta))
    shapes = [array.shape for array in as_arrays]

    if shapes != [(1,), (1,), (1,)]:
        logging.warning("You are trying to give TraceWin an array of eps, "
                        "alpha or beta, while it should be a float. I suspect "
                        "that the current BeamParameters was generated by a "
                        "SimulationOutuput, while it should have been created "
                        "by a ListOfElements (initial beam state). Taking "
                        "first element of each array...")
    return as_arrays[0][0], as_arrays[1][0], as_arrays[2][0]
