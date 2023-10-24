#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather the beam parameters of all the phase spaces.

.. todo::
    May be interesting to create a ``BeamInitialState``, in the same fashion as
    :class:`.ParticleInitialState`. ``BeamInitialState`` would be the only one
    to have a ``tracewin_command``.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
from typing import Any, Callable
from dataclasses import dataclass
import logging

import numpy as np

from core.beam_parameters.single_phase_space_beam_parameters import (
    SinglePhaseSpaceBeamParameters,
    mismatch_single_phase_space,
)
from core.elements.element import Element

from tracewin_utils.interface import beam_parameters_to_command

from util import converters
from util.helper import (recursive_items,
                         recursive_getter,
                         range_vals,
                         )


IMPLEMENTED_PHASE_SPACES = ('zdelta', 'z', 'phiw', 'x', 'y', 't',
                            'phiw99', 'x99', 'y99')  #:


@dataclass
class BeamParameters:
    r"""
    Hold all emittances, envelopes, etc in various planes.

    Attributes
    ----------
    z_abs : np.ndarray | None, optional
        Absolute position in the linac in m. The default is None.
    gamma_kin : np.ndarray | float | None, optional
        Lorentz gamma factor. The default is None.
    beta_kin : np.ndarray | float | None, optional
        Lorentz gamma factor. The default is None. If ``beta_kin`` is not
        provided but ``gamma_kin`` is, ``beta_kin`` is automatically calculated
        at initialisation.
    element_to_index : Callable[[str | Element, str | None],
                                 int | slice] | None, optional
        Takes an :class:`.Element`, its name, ``'first'`` or ``'last'`` as
        argument, and returns corresponding index. Index should be the same in
        all the arrays attributes of this class: ``z_abs``, ``beam_parameters``
        attributes, etc. Used to easily ``get`` the desired properties at the
        proper position. The default is None.
    zdelta : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[z-z\delta]` plane.
    z : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[z-z']` plane.
    phiw : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[\phi-W]` plane.
    x : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[x-x']` plane. Only used with 3D
        simulations.
    y : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[y-y']` plane. Only used with 3D
        simulations.
    t : SinglePhaseSpaceBeamParameters
        Holds beam parameters in the :math:`[t-t']` (transverse) plane. Only
        used with 3D simulations.
    phiw99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the :math:`[\phi-W]` plane. Only used with
        multiparticle simulations.
    x99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the :math:`[x-x']` plane. Only used with
        multiparticle simulations.
    y99 : SinglePhaseSpaceBeamParameters
        Holds 99% beam parameters in the :math:`[y-y']` plane. Only used with
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
            key, phase_space = _separate_var_from_phase_space(key)
            return key in recursive_items(vars(vars(self)[phase_space]))
        return key in recursive_items(vars(self))

    def get(self, *keys: str, to_numpy: bool = True, none_to_nan: bool = False,
            elt: Element | None = None, pos: str | None = None,
            phase_space: str | None = None, **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Notes
        -----
        What is particular in this getter is that all
        :class:`SinglePhaseSpaceBeamParameters` attributes have attributes with
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
        for key in keys:
            short_key = key
            if _phase_space_name_hidden_in_key(key):
                if phase_space is not None:
                    logging.warning(
                        "Ambiguous: you asked for two phase-spaces. One with "
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
            understandable by :meth:`SinglePhaseSpaceBeamParameters.__init__`:
            ``'alpha'``, ``'beta'``, ``'gamma'``, ``'eps'``, ``'twiss'``,
            ``'envelope_pos'`` and ``'envelope_energy'`` are allowed values.

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
        r"""Create the desired longitudinal planes from :math:`[z-\delta]`."""
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
