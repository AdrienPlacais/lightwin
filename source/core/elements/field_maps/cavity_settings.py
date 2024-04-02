#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store cavity settings that can change during an optimisation.

.. note::
    As for now, :class:`.FieldMap` is the only object to have its properties in
    a dedicated object.

.. todo::
    Check behavior with rephased

"""
import cmath
from collections.abc import Callable
from functools import partial
import logging
import math
from typing import TypeVar

from scipy.optimize import NonlinearConstraint, minimize_scalar

from util.phases import (
    diff_angle,
    phi_0_abs_to_rel,
    phi_0_rel_to_abs,
    phi_bunch_to_phi_rf,
)


ALLOWED_REFERENCES = ('phi_0_abs', 'phi_0_rel', 'phi_s')  #:
# warning: doublon with field_map.IMPLEMENTED_STATUS
ALLOWED_STATUS = ('nominal',
                  'rephased (in progress)',
                  'rephased (ok)',
                  'failed',
                  'compensate (in progress)',
                  'compensate (ok)',
                  'compensate (not ok)'
                  )  #:


def compute_cavity_parameters(integrated_field: complex
                              ) -> tuple[float, float]:
    """Compute the synchronous phase and accelerating voltage."""
    polar = cmath.polar(integrated_field)
    v_cav, phi_s = polar[0], polar[1]
    return v_cav, phi_s


class CavitySettings:
    """Hold the cavity parameters that can vary during optimisation.

    .. todo::
        Maybe set deletters?

    .. todo::
        Which syntax for when I want to compute the value of a property but not
        return it? Maybe a ``_ = self.phi_0_abs``? Maybe this case should not
        appear here, appart for when I debug.

    .. note::
        In this routine, all phases are defined in radian and are rf phases.

    .. todo::
        Determine if status should be kept here or in the field map.

    .. todo::
        For TraceWin solver, I will also need the field map index.

    """

    def __init__(self,
                 k_e: float,
                 phi: float,
                 reference: str,
                 status: str,
                 freq_bunch_mhz: float,
                 freq_cavity_mhz: float | None = None,
                 transf_mat_func_wrappers: dict[str, Callable] | None = None,
                 ) -> None:
        """Instantiate the object.

        Parameters
        ----------
        k_e : float
            Amplitude of the electric field.
        phi : float
            Input phase. Must be absolute or relative entry phase, or
            synchronous phase.
        reference : {'phi_0_abs', 'phi_0_rel', 'phi_s'}
            Name of the phase used for reference. When a particle enters the
            cavity, this is the phase that is not recomputed.
        status : str
            A value in :var:`ALLOWED_STATUS`.
        freq_bunch_mhz : float
            Bunch frequency in MHz.
        freq_cavity_mhz : float | None, optional
            Frequency of the cavity in MHz. The default is None, which happens
            when the :class:`.ListOfElements` is under creation and we did not
            process the ``FREQ`` commands yet.

        """
        self.k_e = k_e
        self._reference: str
        self.reference = reference

        self._phi_0_abs: float | None
        self._phi_0_rel: float | None
        self._phi_s: float | None
        self._v_cav_mv: float | None
        self._phi_rf: float
        self._set_reference_phase(phi)

        self.status = status

        self.transf_mat_func_wrappers: dict[str, Callable] = {}
        if transf_mat_func_wrappers is not None:
            self.transf_mat_func_wrappers = transf_mat_func_wrappers
        self._phi_0_rel_to_phi_s: Callable
        self._phi_s_to_phi_0_rel: Callable

        self._freq_bunch_mhz = freq_bunch_mhz
        self._bunch_phase_to_rf_phase: Callable[[float], float]
        self.freq_cavity_mhz: float
        if freq_cavity_mhz is not None:
            self.set_bunch_to_rf_freq_func(freq_cavity_mhz)

    def __str__(self) -> str:
        """Print out the different phases/k_e, and which one is the reference.

        .. note::
            ``None`` means that the phase was not calculated.

        """
        out = f"Status: {self.status:>10} | "
        out += f"Reference: {self.reference:>10} | "
        phases_as_string = [
            self._attr_to_str(phase_name)
            for phase_name in ('_phi_0_abs', '_phi_0_rel', '_phi_s', 'k_e')]
        return out + ' | '.join(phases_as_string)

    def __repr__(self) -> str:
        """Return the same thing as str."""
        return str(self)

    def _attr_to_str(self, attr_name: str) -> str:
        """Give the attribute as string."""
        attr_val = getattr(self, attr_name, None)
        if attr_val is None:
            return f"{attr_name}: {'None':>7}"
        return f"{attr_name}: {attr_val:3.5f}"

    def _set_reference_phase(self, phi: float) -> None:
        """Update the reference phase.

        Also delete the other ones that are now outdated to avoid any
        confusion.

        """
        self._delete_non_reference_phases()
        setattr(self, self.reference, phi)

    def _delete_non_reference_phases(self) -> None:
        """Reset the phases that are not the reference to None."""
        if self.reference == 'phi_0_abs':
            self._phi_0_rel = None
            self._phi_s = None
            return
        if self.reference == 'phi_0_rel':
            self._phi_0_abs = None
            self._phi_s = None
            return
        if self.reference == 'phi_s':
            self._phi_0_abs = None
            self._phi_0_rel = None
            return
        raise ValueError(f"{self.reference = } not implemented.")

    def _check_consistency_of_status_and_reference(self) -> None:
        """Perform some tests on ``status`` and ``reference``.

        .. todo::
            Maybe not necessary to raise an error when there is a mismatch.

        """
        if 'rephased' in self.status:
            assert self.reference == 'phi_0_rel'

    def set_bunch_to_rf_freq_func(self, freq_cavity_mhz: float) -> None:
        """Use cavity frequency to set a bunch -> rf freq function.

        This method is called by the :class:`.Freq`.

        Parameters
        ----------
        freq_cavity_mhz : float
            Frequency in the cavity in MHz.

        """
        self.freq_cavity_mhz = freq_cavity_mhz
        bunch_phase_to_rf_phase = partial(
            phi_bunch_to_phi_rf,
            freq_cavity_mhz / self._freq_bunch_mhz)
        self._bunch_phase_to_rf_phase = bunch_phase_to_rf_phase

# =============================================================================
# Reference
# =============================================================================
    @property
    def reference(self) -> str:
        """Say what is the reference phase.

        .. list-table:: Equivalents of ``reference`` in TraceWin's \
                ``FIELD_MAP``
            :widths: 50, 50
            :header-rows: 1

            * - LightWin's ``reference``
              - TraceWin
            * - ``'phi_0_rel'``
              - ``P = 0``
            * - ``'phi_0_abs'``
              - ``P = 1``
            * - ``'phi_s'``
              - ``SET_SYNC_PHASE``

        """
        return self._reference

    @reference.setter
    def reference(self, value: str) -> None:
        """Set the value of reference, check that it is valid."""
        assert value in ALLOWED_REFERENCES
        self._reference = value

# =============================================================================
# Status
# =============================================================================
    @property
    def status(self) -> str:
        """Give the status of the cavity under study."""
        return self._status

    @status.setter
    def status(self, value: str) -> None:
        """Check that new status is allowed, set it.

        Also checks consistency between the value of the new status and the
        value of the :attr:`.reference`.

        .. todo::
            logging.warning("Check that beam_calc_param is still updated."
                        "As in FieldMap.update_status")

        """
        assert value in ALLOWED_STATUS
        self._status = value
        if value == 'failed':
            self.k_e = 0.

        # logging.warning("Check that beam_calc_param is still updated."
            # "As in FieldMap.update_status")
        # this function changes the transfer matrix function, and the function
        # that converts results to a dictionary for EnvelopeiD. Does nothing
        # with TraceWin.

        self._check_consistency_of_status_and_reference()

# =============================================================================
# Absolute phi_0
# =============================================================================
    @property
    def phi_0_abs(self) -> None:
        """Declare the absolute entry phase property."""

    @phi_0_abs.setter
    def phi_0_abs(self, value: float) -> None:
        """Set the absolute entry phase."""
        self._phi_0_abs = value

    @phi_0_abs.getter
    def phi_0_abs(self) -> float | None:
        """Get the absolute entry phase, compute if necessary."""
        if self._phi_0_abs is not None:
            return self._phi_0_abs

        if not hasattr(self, '_phi_rf'):
            return None

        if self._phi_0_rel is not None:
            self.phi_0_abs = phi_0_rel_to_abs(self._phi_0_rel, self._phi_rf)
            return self._phi_0_abs

        logging.error("The phase was not initialized. Returning None...")
        return None

# =============================================================================
# Relative phi_0
# =============================================================================
    @property
    def phi_0_rel(self) -> None:
        """Get relative entry phase, compute it if necessary."""

    @phi_0_rel.setter
    def phi_0_rel(self, value: float) -> None:
        """Set the relative entry phase."""
        self._phi_0_rel = value

    @phi_0_rel.getter
    def phi_0_rel(self) -> float | None:
        """Get the relative entry phase, compute it if necessary."""
        if self._phi_0_rel is not None:
            return self._phi_0_rel

        if not hasattr(self, '_phi_rf'):
            return None

        if self._phi_0_abs is not None:
            self.phi_0_rel = phi_0_abs_to_rel(self._phi_0_abs, self._phi_rf)
            return self._phi_0_rel

        if self._phi_s is None:
            logging.error("No phase was initialized. Returning None...")
            return None

        phi_0_from_phi_s_calc = getattr(self, '_phi_s_to_phi_0_rel', None)
        if phi_0_from_phi_s_calc is None:
            logging.error("You must set a function to compute phi_0_rel from "
                          "phi_s with CavitySettings.set_phi_s_calculators"
                          " method.")
            return None

        self.phi_0_rel = phi_0_from_phi_s_calc(self._phi_s)
        return self._phi_0_rel

# =============================================================================
# Synchronous phase
# =============================================================================
    @property
    def phi_s(self) -> None:
        """Get synchronous phase, compute it if necessary."""

    @phi_s.setter
    def phi_s(self, value: float) -> None:
        """Set the synchronous phase to desired value."""
        self._phi_s = value

    @phi_s.getter
    def phi_s(self) -> float | None:
        """Get the synchronous phase, and compute it if necessary.

        .. note::
            It is mandatory for the calculation of this quantity to compute
            propagation of the particle in the cavity.

        See Also
        --------
        set_phi_s_calculators

        """
        if self._phi_s is not None:
            return self._phi_s

        if not hasattr(self, '_phi_rf'):
            return None

        # We omit the _ in front of phi_0_rel to compute it if necessary
        if self.phi_0_rel is None:
            logging.error("You must declare the particle entry phase in the "
                          "cavity to compute phi_0_rel and then phi_s.")
            return None

        phi_s_calc = getattr(self, '_phi_0_rel_to_phi_s', None)
        if phi_s_calc is None:
            logging.error("You must set a function to compute phi_s from "
                          "phi_0_rel with CavitySettings.set_phi_s_calculators"
                          " method.")
            return None

        self._phi_s = phi_s_calc(self.phi_0_rel)
        return self._phi_s

    def set_phi_s_calculators(self,
                              solver_id: str,
                              w_kin: float,
                              **kwargs) -> None:
        """Set the functions that compute synchronous phase.

        This function must be called every time the kinetic energy at the
        entrance of the cavity is changed (like this occurs during optimisation
        process) or when the synchronous phase must be calculated with another
        solver.

        See Also
        --------
        set_beam_calculator

        """
        if 'phi_0_rel' in kwargs:
            del kwargs['phi_0_rel']
        transf_mat_function_wrapper = self.transf_mat_func_wrappers.get(
            solver_id, None)
        if transf_mat_function_wrapper is None:
            logging.error(f"No function to compute beam propagation matching "
                          f"{solver_id = } was found. You must set it with "
                          "CavitySettings.set_beam_calculator.")
            return None

        def phi_0_rel_to_phi_s(phi_0_rel: float) -> float:
            """Compute propagation of the beam, deduce synchronous phase."""
            results = transf_mat_function_wrapper(phi_0_rel=phi_0_rel,
                                                  w_kin_in=w_kin,
                                                  **kwargs)
            phi_s = results['cav_params']['phi_s']
            return phi_s

        def _residue_func(phi_0_rel: float, phi_s: float) -> float:
            """Compute difference between given and calculated ``phi_s``."""
            calculated_phi_s = phi_0_rel_to_phi_s(phi_0_rel)
            residue = diff_angle(phi_s, calculated_phi_s)
            return residue**2

        def phi_s_to_phi_0_rel(phi_s: float) -> float:
            """Call recursively ``phi_0_rel_to_phi_s`` to find ``phi_s``."""
            out = minimize_scalar(_residue_func,
                                  bounds=(0., 2. * math.pi),
                                  args=(phi_s, ))
            if not out.success:
                logging.error('Synch phase not found')
            return out.x

        self._phi_0_rel_to_phi_s = phi_0_rel_to_phi_s
        self._phi_s_to_phi_0_rel = phi_s_to_phi_0_rel

    def set_beam_calculator(self,
                            solver_id: str,
                            transf_mat_function_wrapper: Callable) -> None:
        """Add or modify a function to compute beam propagation.

        Must be called at the creation of the corresponding
        :class:`.ElementBeamCalculatorParameters` to compute synchronous
        phases.

        """
        self.transf_mat_func_wrappers[solver_id] = transf_mat_function_wrapper

# =============================================================================
# Phase of synchronous particle
# =============================================================================
    @property
    def phi_rf(self) -> None:
        """Declare the synchronous particle entry phase."""

    @phi_rf.setter
    def phi_rf(self, value: float) -> None:
        """Set the new synch particle entry phase, remove value to update.

        We also remove the synchronous phase. In most of the situations, we
        also remove ``phi_0_rel`` and keep ``phi_0_abs`` (we must ensure that
        ``phi_0_abs`` was previously set).
        The exception is when the cavity has the ``'rephased'`` status. In this
        case, we keep the relative ``phi_0`` and absolute ``phi_0`` will be
        recomputed when/if it is called.

        Parameters
        ----------
        value : float
            New rf phase of the synchronous particle at the entrance of the
            cavity.

        """
        self._phi_rf = value
        self._delete_non_reference_phases()

        # if self.status == 'rephased (in progress)':
        #     self.phi_0_rel
        #     self._phi_0_abs = None
        #     return
        # self.phi_0_abs
        # self._phi_0_rel = None

    @phi_rf.getter
    def phi_rf(self) -> float:
        """Get the rf phase of synch particle at entrance of cavity."""
        return self._phi_rf

    @property
    def phi_bunch(self) -> None:
        """Declare the synchronous particle entry phase in bunch.

        .. note::
            Cannot access this property, as I think it is not useful. It is
            only used to set the rf phase. Will be changed in the future if
            necessary.

        """

    @phi_bunch.setter
    def phi_bunch(self, value: float) -> None:
        """Convert bunch to rf frequency."""
        self.phi_rf = self._bunch_phase_to_rf_phase(value)

    @phi_bunch.getter
    def phi_bunch(self) -> None:
        """Raise an error, as accessing this property should not be needed."""
        raise IOError("Why do you need to access phi_bunch??")

    # .. list-table:: Meaning of status
    #     :widths: 40, 60
    #     :header-rows: 1

    #     * - LightWin's ``status``
    #       - Meaning
    #     * - ``'nominal'``
    #       - ``phi_0`` and ``k_e`` match the original ``.dat`` file
    #     * - ``'rephased (in progress)'``
    #       - ``k_e`` unchanged, trying to find the proper ``phi_0`` (usually,\
    #       just keep the ``phi_0_rel``)
    #     * - ``'rephased (ok)'``
    #       - ``k_e`` unchanged, new ``phi_0`` (usually ``phi_0_rel`` is
    #         unchanged)
    #     * - ``'failed'``
    #       - ``k_e = 0``
    #     * - ``'compensate (in progress)'``
    #       - trying to find new ``k_e`` and ``phi_0``
    #     * - ``'compensate (ok)'``
    #       - new ``k_e`` and ``phi_0`` were found, optimisation algorithm is
    #         happy with it
    #     * - ``'compensate (not ok)'``
    #       - new ``k_e`` and ``phi_0`` were found, optimisation algorithm is
    #         not happy with it


# @dataclass
# class OldCavitySettings:
    # """Settings of a single cavity."""

    # cavity: FieldMap
    # index: int
    # k_e: float | None = None
    # phi_0_abs: float | None = None
    # phi_0_rel: float | None = None
    # phi_s: float | None = None

    # def __post_init__(self):
        # """Test that only one phase was given."""
        # if not self._is_valid_phase_input:
            # logging.error("You gave CavitySettings several phases... "
                          # "Which one should it take? Ignoring phases.")
            # self.phi_0_abs = None
            # self.phi_0_rel = None
            # self.phi_s = None

    # def tracewin_command(self, delta_phi_bunch: float = 0.) -> list[str]:
        # """Call the function from ``tracewin_utils`` to modify TW call."""
        # phi_0_abs = self._tracewin_phi_0_abs(delta_phi_bunch)
        # tracewin_command = single_cavity_settings_to_command(self.index,
                                                             # phi_0_abs,
                                                             # self.k_e)
        # return tracewin_command

    # def has(self, key: str) -> bool:
        # """Tell if the required attribute is in this class."""
        # return key in recursive_items(vars(self))

    # def get(self, *keys: str, to_deg: bool = False, **kwargs: dict
            # ) -> tuple[Any]:
        # """Shorthand to get attributes."""
        # val: dict[str, Any] = {}
        # for key in keys:
            # val[key] = []

        # for key in keys:
            # if not self.has(key):
                # val[key] = None
                # continue

            # val[key] = recursive_getter(key, vars(self), **kwargs)

            # if val[key] is not None and to_deg and 'phi' in key:
                # val[key] = np.rad2deg(val[key])

        # out = [val[key] for key in keys]
        # if len(out) == 1:
            # return out[0]

        # return tuple(out)

    # @property
    # def _is_valid_phase_input(self) -> bool:
        # """Assert that no more than one phase was given as input."""
        # phases = [self.phi_0_abs, self.phi_0_rel, self.phi_s]
        # number_of_given_phases = sum(1 for phase in phases
                                     # if phase is not None)
        # if number_of_given_phases > 1:
            # return False
        # return True

    # def _tracewin_phi_0_abs(self, delta_phi_bunch: float) -> float:
        # """
        # Return the proper absolute entry phase of the cavity.

        # The attribute `self.phi_0_abs` is only valid if the beam has a null
        # absolute phase at the entry of the linac. When working with `TraceWin`,
        # the beam has an absolute phase at the entry of the compensation zone.
        # Hence we compute the new `phi_0_abs` that will be properly taken into
        # account.

        # Three figure cases, according to the nature of the input:
            # - phi_0_rel
            # - phi_0_abs (phi_abs = 0. at entry of `ListOfElements`)
            # - phi_s

        # Parameters
        # ----------
        # delta_phi_bunch : float
            # Difference between the absolute entry phase of the `ListOfElements`
            # under study and the entry phase of the `ListOfElements` for which
            # given phi_0_abs is valid.

        # Returns
        # -------
        # phi_0 : float
            # New absolute phase.

        # """
        # if not self._is_valid_phase_input:
            # logging.error("More than one phase was given.")

        # if self.phi_0_abs is not None:
            # phi_0_abs = phi_0_abs_with_new_phase_reference(
                # self.phi_0_abs,
                # delta_phi_bunch * self.cavity.rf_field.n_cell)
            # return phi_0_abs

        # if self.phi_0_rel is not None:
            # raise NotImplementedError("TraceWin switch abs/rel not supported.")

        # if self.phi_s is not None:
            # raise NotImplementedError("TraceWin fit on phi_s not supported.")
