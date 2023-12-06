#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gather the beam parameters of all the phase spaces.

For a list of the units associated with every parameter, see
:ref:`units-label`.

"""
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Self

import numpy as np

from core.beam_parameters.initial_beam_parameters import (
    InitialBeamParameters,
    phase_space_name_hidden_in_key,
    separate_var_from_phase_space)
from core.beam_parameters.phase_space.phase_space_beam_parameters import (
    PhaseSpaceBeamParameters)
from core.elements.element import Element
from tracewin_utils.interface import beam_parameters_to_command


@dataclass
class BeamParameters(InitialBeamParameters):
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
    sigma_in : np.ndarray | None, optional
        Holds the (6, 6) :math:`\sigma` beam matrix at the entrance of the
        linac/portion of linac. The default is None.
    zdelta, z, phiw, x, y, t : PhaseSpaceBeamParameters
        Holds beam parameters respectively in the :math:`[z-z\delta]`,
        :math:`[z-z']`, :math:`[\phi-W]`, :math:`[x-x']`, :math:`[y-y']` and
        :math:`[t-t']` planes.
    phiw99, x99, y99 : PhaseSpaceBeamParameters
        Holds 99% beam parameters respectively in the :math:`[\phi-W]`,
        :math:`[x-x']` and :math:`[y-y']` planes. Only used with multiparticle
        simulations.
    element_to_index : Callable[[str | Element, str | None], int | slice]
        Takes an :class:`.Element`, its name, ``'first'`` or ``'last'`` as
        argument, and returns corresponding index. Index should be the same in
        all the arrays attributes of this class: ``z_abs``, ``beam_parameters``
        attributes, etc. Used to easily ``get`` the desired properties at the
        proper position.

    """

    # Override type from mother class
    z_abs: np.ndarray
    gamma_kin: np.ndarray
    beta_kin: np.ndarray
    sigma_in: np.ndarray | None = None

    element_to_index: Callable[[str | Element, str | None], int | slice] \
        = lambda _elt, _pos: slice(0, -1)

    def __post_init__(self) -> None:
        """Declare the phase spaces."""
        self.n_points = np.atleast_1d(self.z_abs).shape[0]

        # Override types from mother class
        self.zdelta: PhaseSpaceBeamParameters
        self.z: PhaseSpaceBeamParameters
        self.phiw: PhaseSpaceBeamParameters
        self.x: PhaseSpaceBeamParameters
        self.y: PhaseSpaceBeamParameters
        self.t: PhaseSpaceBeamParameters
        self.phiw99: PhaseSpaceBeamParameters
        self.x99: PhaseSpaceBeamParameters
        self.y99: PhaseSpaceBeamParameters

    def get(self,
            *keys: str,
            to_numpy: bool = True,
            none_to_nan: bool = False,
            elt: Element | None = None,
            pos: str | None = None,
            phase_space_name: str | None = None,
            **kwargs: Any) -> Any:
        """
        Shorthand to get attributes from this class or its attributes.

        Notes
        -----
        What is particular in this getter is that all
        :class:`.PhaseSpaceBeamParameters` attributes have attributes with
        the same name: ``twiss``, ``alpha``, ``beta``, ``gamma``, ``eps``,
        ``envelope_pos`` and ``envelope_energy``.

        Hence, you must provide either a ``phase_space`` argument which shall
        be in :data:`IMPLEMENTED_PHASE_SPACES`, either you must append the
        name of the phase space to the name of the desired variable with an
        underscore.

        See Also
        --------
        :meth:`has`

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
        phase_space_name : str | None, optional
            Phase space in which you want the key. The default is None. In this
            case, the quantities from the ``zdelta`` phase space are taken.
            Otherwise, it must be in :data:`IMPLEMENTED_PHASE_SPACES`.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        assert 'phase_space' not in kwargs
        val = {key: [] for key in keys}

        # Explicitely look into a specific PhaseSpaceBeamParameters
        if phase_space_name is not None:
            phase_space = getattr(self, phase_space_name)
            val = {key: getattr(phase_space, key) for key in keys}

        else:
            for key in keys:
                if phase_space_name_hidden_in_key(key):
                    short_key, phase_space_name = \
                        separate_var_from_phase_space(key)
                    assert hasattr(self, phase_space_name), \
                        f"{phase_space_name = } not set for current "\
                        + "BeamParameters object."
                    phase_space = getattr(self, phase_space_name)
                    val[key] = getattr(phase_space, short_key)
                    continue

                # Look for key in BeamParameters
                if self.has(key):
                    val[key] = getattr(self, key)
                    continue

                val[key] = None

        if elt is not None:
            idx = self.element_to_index(elt=elt, pos=pos)
            val = {_key: _value[idx]
                   if _value is not None else None
                   for _key, _value in val.items()
                   }

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
    def sigma(self) -> np.ndarray:
        """Give value of sigma."""
        warnings.warn("Will be deprecated, unless there is a need for this",
                      FutureWarning)
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

    def sub_sigma_in(self,
                     phase_space_name: str,
                     ) -> np.ndarray:
        r"""Give the entry :math:`\sigma` beam matrix in a single phase space.

        Parameters
        ----------
        phase_space_name : {'x', 'y', 'zdelta'}
            Name of the phase space from which you want the :math:`\sigma` beam
            matrix.

        Returns
        -------
        sigma : np.ndarray
            ``(2, 2)`` :math:`\sigma` beam matrix at the linac entrance, in a
            single phase space.

        """
        assert self.sigma_in is not None
        if phase_space_name == 'x':
            return self.sigma_in[:2, :2]
        if phase_space_name == 'y':
            return self.sigma_in[2:4, 2:4]
        if phase_space_name == 'zdelta':
            return self.sigma_in[4:, 4:]
        raise IOError(f"{phase_space_name = } is not allowed.")

    @property
    def tracewin_command(self) -> list[str]:
        """Return the proper input beam parameters command."""
        logging.critical("is this method still used??")
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

    def set_mismatches(self,
                       reference_beam_parameters: Self,
                       *phase_space_names: str,
                       **mismatch_kw: bool) -> None:
        """Compute and set mismatch in every possible phase space."""
        z_abs = self.z_abs
        reference_z_abs = reference_beam_parameters.z_abs

        phase_space, reference_phase_space = None, None
        for phase_space_name in phase_space_names:
            if phase_space_name == 't':
                self._set_mismatch_for_transverse(**mismatch_kw)
                continue
            phase_space, reference_phase_space = self._get_phase_spaces(
                reference_beam_parameters,
                phase_space_name,
                **mismatch_kw)
            if reference_phase_space is None or phase_space is None:
                continue
            phase_space.set_mismatch(reference_phase_space,
                                     z_abs,
                                     reference_z_abs,
                                     **mismatch_kw)

    def _get_phase_spaces(self,
                          reference_beam_parameters: Self,
                          phase_space_name: str,
                          raise_missing_phase_space_error: bool,
                          **mismatch_kw: bool,
                          ) -> tuple[PhaseSpaceBeamParameters | None,
                                     PhaseSpaceBeamParameters | None]:
        """Get the two phase spaces between which mismatch will be computed."""
        if not hasattr(self, phase_space_name):
            if raise_missing_phase_space_error:
                raise IOError(f"Phase space {phase_space_name} not "
                              "defined in fixed linac. Cannot compute "
                              "mismatch.")
            return None, None

        if not hasattr(reference_beam_parameters, phase_space_name):
            if raise_missing_phase_space_error:
                raise IOError(f"Phase space {phase_space_name} not "
                              "defined in reference linac. Cannot compute "
                              "mismatch.")
            return None, None

        phase_space = getattr(self, phase_space_name)
        reference_phase_space = getattr(reference_beam_parameters,
                                        phase_space_name)
        return phase_space, reference_phase_space

    def _set_mismatch_for_transverse(
            self,
            raise_missing_phase_space_error: bool = True,
            raise_missing_mismatch_error: bool = True,
            **mismatch_kw: bool) -> None:
        """Set ``t`` mismatch as average of ``x`` and ``y``."""
        if not hasattr(self, 'x'):
            if raise_missing_phase_space_error:
                raise IOError("Phase space x not defined in fixed linac. "
                              "Cannot compute transverse mismatch.")
            return None

        if not hasattr(self, 'y'):
            if raise_missing_phase_space_error:
                raise IOError("Phase space y not defined in fixed linac. "
                              "Cannot compute transverse mismatch.")
            return None

        if not hasattr(self.x, "mismatch_factor"):
            if raise_missing_mismatch_error:
                raise IOError("Phase space x has no calculated mismatch. "
                              "Cannot compute transverse mismatch.")
            return None

        if not hasattr(self.y, "mismatch_factor"):
            if raise_missing_mismatch_error:
                raise IOError("Phase space y has no calculated mismatch. "
                              "Cannot compute transverse mismatch.")
            return None

        self.t.mismatch_factor = .5 * (self.x.mismatch_factor
                                       + self.y.mismatch_factor)


# =============================================================================
# Private
# =============================================================================
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
