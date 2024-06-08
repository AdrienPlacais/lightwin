"""Store cavity settings that can change during an optimisation.

.. note::
    As for now, :class:`.FieldMap` is the only object to have its properties in
    a dedicated object.

.. todo::
    Similar to synchronous phase, allow for V_cav to be "master" instead of
    k_e.

See Also
--------
NewRfField

"""

import logging
import math
from collections.abc import Callable
from functools import partial
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import minimize_scalar

from util.phases import (
    diff_angle,
    phi_0_abs_to_rel,
    phi_0_rel_to_abs,
    phi_bunch_to_phi_rf,
    phi_rf_to_phi_bunch,
)

REFERENCE_T = Literal["phi_0_abs", "phi_0_rel", "phi_s"]
ALLOWED_REFERENCES = ("phi_0_abs", "phi_0_rel", "phi_s")  #:
# warning: doublon with field_map.IMPLEMENTED_STATUS
STATUS_T = Literal[
    "nominal",
    "rephased (in progress)",
    "rephased (ok)",
    "failed",
    "compensate (in progress)",
    "compensate (ok)",
    "compensate (not ok)",
]
ALLOWED_STATUS = (
    "nominal",
    "rephased (in progress)",
    "rephased (ok)",
    "failed",
    "compensate (in progress)",
    "compensate (ok)",
    "compensate (not ok)",
)  #:


class CavitySettings:
    """Hold the cavity parameters that can vary during optimisation.

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

    def __init__(
        self,
        k_e: float,
        phi: float,
        reference: REFERENCE_T,
        status: STATUS_T,
        freq_bunch_mhz: float,
        freq_cavity_mhz: float | None = None,
        transf_mat_func_wrappers: dict[str, Callable] | None = None,
        phi_s_funcs: dict[str, Callable] | None = None,
    ) -> None:
        """Instantiate the object.

        Parameters
        ----------
        k_e : float
            Amplitude of the electric field.
        phi : float
            Input phase. Must be absolute or relative entry phase, or
            synchronous phase.
        reference : REFERENCE_T
            Name of the phase used for reference. When a particle enters the
            cavity, this is the phase that is not recomputed.
        status : STATUS_T
            A value in :var:`ALLOWED_STATUS`.
        freq_bunch_mhz : float
            Bunch frequency in MHz.
        freq_cavity_mhz : float | None, optional
            Frequency of the cavity in MHz. The default is None, which happens
            when the :class:`.ListOfElements` is under creation and we did not
            process the ``FREQ`` commands yet.
        transf_mat_func_wrappers : dict[str, Callable] | None, optional
            A dictionary which keys are the different :class:`.BeamCalculator`
            ids, and values are corresponding functions to compute propagation
            of the beam. The default is None, in which case attribute is not
            set.
        phi_s_funcs : dict[str, Callable] | None, optional
            A dictionary which keys are the different :class:`.BeamCalculator`
            ids, and values are corresponding functions to compute synchronous
            phase and accelerating voltage from the ouput of corresponding
            ``transf_mat_func_wrapper``. The default is None, in which case
            attribute is not set.

        """
        self.k_e = k_e
        self._reference: REFERENCE_T
        self.reference = reference
        self.phi_ref = phi

        self._phi_0_abs: float
        self._phi_0_rel: float
        self._phi_s: float
        self._v_cav_mv: float
        self._phi_rf: float
        self._phi_bunch: float

        self._status: STATUS_T
        self.status = status

        self.transf_mat_func_wrappers: dict[str, Callable] = {}
        if transf_mat_func_wrappers is not None:
            self.transf_mat_func_wrappers = transf_mat_func_wrappers
        self.phi_s_funcs: dict[str, Callable] = {}
        if phi_s_funcs is not None:
            self.phi_s_funcs = phi_s_funcs
        self._phi_0_rel_to_cavity_parameters: Callable
        self._phi_s_to_phi_0_rel: Callable

        self._freq_bunch_mhz = freq_bunch_mhz
        self.bunch_phase_to_rf_phase: Callable[[float], float]
        self.rf_phase_to_bunch_phase: Callable[[float], float]
        self.freq_cavity_mhz: float
        self.omega0_rf: float
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
            for phase_name in ("_phi_0_abs", "_phi_0_rel", "_phi_s", "k_e")
        ]
        return out + " | ".join(phases_as_string)

    def __repr__(self) -> str:
        """Return the same thing as str."""
        return str(self)

    def __eq__(self, other: Self) -> bool:  # type: ignore
        """Check if two cavity settings are identical."""
        check = (
            self.k_e == other.k_e
            and self.phi_ref == other.phi_ref
            and self.reference == other.reference
        )
        # also check for phi_bunch?
        return check

    @classmethod
    def from_other_cavity_settings(
        cls,
        other: Self,
        reference: REFERENCE_T | None = None,
    ) -> Self:
        """Create settings with same settings as provided."""
        if reference is None:
            reference = other.reference
        assert reference is not None
        settings = cls(
            other.k_e,
            getattr(other, reference),
            reference,
            other.status,
            other._freq_bunch_mhz,
            other.freq_cavity_mhz,
            transf_mat_func_wrappers=other.transf_mat_func_wrappers,
            phi_s_funcs=other.phi_s_funcs,
        )
        return settings

    @classmethod
    def from_optimisation_algorithm(
        cls,
        base: Self,
        k_e: float,
        phi: float,
        status: STATUS_T,
        reference: REFERENCE_T | None = None,
    ) -> Self:
        """Create settings based on ``base`` with different ``k_e``, ``phi_0``.

        Parameters
        ----------
        base : Self
            The reference :class:`CavitySettings`. A priori, this is the
            nominal settings.
        k_e : float
            New field amplitude.
        phi : float
            New reference phase. Its nature is defined by ``reference``.
        status : {"compensate (in progress)", "compensate (ok)", "compensate \
                (not ok)"}
            Status of the created settings.
        reference : {"phi_0_rel", "phi_0_abs", "phi_s"}
            The phase used as a reference.

        Returns
        -------
        Self
            A new :class:`CavitySettings` with modified amplitude and phase.

        """
        if reference is None:
            reference = base.reference
        assert reference is not None
        settings = cls(
            k_e,
            phi,
            reference,
            status,
            base._freq_bunch_mhz,
            base.freq_cavity_mhz,
            base.transf_mat_func_wrappers,
            base.phi_s_funcs,
        )
        return settings

    def _attr_to_str(self, attr_name: str, to_deg: bool = True) -> str:
        """Give the attribute as string."""
        attr_val = getattr(self, attr_name, None)
        if attr_val is None:
            return f"{attr_name}: {'None':>7}"
        if to_deg and "phi" in attr_name:
            attr_val = math.degrees(attr_val)
            if attr_val > 180.0:
                attr_val -= 360.0
        return f"{attr_name}: {attr_val:3.5f}"

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return hasattr(self, key)

    def get(
        self, *keys: str, to_deg: bool = False, **kwargs: bool | str | None
    ) -> Any:
        """Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys : str
            Name of the desired attributes.
        **kwargs : bool | str | None
            Other arguments passed to recursive getter.

        Returns
        -------
        out : Any
            Attribute(s) value(s).

        """
        val: dict[str, Any] = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            val[key] = getattr(self, key)
            if to_deg and "phi" in key:
                val[key] = math.degrees(val[key])

        out = [val[key] for key in keys]
        if len(keys) == 1:
            return out[0]
        return tuple(out)

    def _check_consistency_of_status_and_reference(self) -> None:
        """Perform some tests on ``status`` and ``reference``.

        .. todo::
            Maybe not necessary to raise an error when there is a mismatch.

        """
        if "rephased" in self.status:
            assert self.reference == "phi_0_rel"

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
            phi_bunch_to_phi_rf, freq_cavity_mhz / self._freq_bunch_mhz
        )
        self.bunch_phase_to_rf_phase = bunch_phase_to_rf_phase

        rf_phase_to_bunch_phase = partial(
            phi_rf_to_phi_bunch, self._freq_bunch_mhz / freq_cavity_mhz
        )
        self.rf_phase_to_bunch_phase = rf_phase_to_bunch_phase

        self.omega0_rf = 2e6 * math.pi * freq_cavity_mhz

    # =============================================================================
    # Reference
    # =============================================================================
    @property
    def reference(self) -> REFERENCE_T:
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
    def reference(self, value: REFERENCE_T) -> None:
        """Set the value of reference, check that it is valid."""
        assert value in ALLOWED_REFERENCES
        self._reference = value

    @property
    def phi_ref(self) -> None:
        """Declare a shortcut to the reference entry phase."""

    @phi_ref.setter
    def phi_ref(self, value: float) -> None:
        """Update the value of the reference entry phase.

        Also delete the other ones that are now outdated to avoid any
        confusion.

        """
        self._delete_non_reference_phases()
        setattr(self, self.reference, value)

    @phi_ref.getter
    def phi_ref(self) -> float:
        """Give the reference phase."""
        phi = getattr(self, self.reference)
        assert isinstance(phi, float)
        return phi

    def _delete_non_reference_phases(self) -> None:
        """Reset the phases that are not the reference to None."""
        for phase in ALLOWED_REFERENCES:
            if phase == self.reference:
                continue
            delattr(self, phase)

    # =============================================================================
    # Status
    # =============================================================================
    @property
    def status(self) -> STATUS_T:
        """Give the status of the cavity under study."""
        return self._status

    @status.setter
    def status(self, value: STATUS_T) -> None:
        """Check that new status is allowed, set it.

        Also checks consistency between the value of the new status and the
        value of the :attr:`.reference`.

        .. todo::
            Check that beam_calc_param is still updated. As in
            FieldMap.update_status

        .. todo::
            As for now: do not update the status directly, prefer calling the
            :meth:`.FieldMap.update_status`

        """
        assert value in ALLOWED_STATUS
        self._status = value
        if value == "failed":
            self.k_e = 0.0
            self.phi_s = np.NaN
            self.v_cav_mv = np.NaN

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

        if not hasattr(self, "_phi_rf"):
            return None

        if self._phi_0_rel is not None:
            self.phi_0_abs = phi_0_rel_to_abs(self._phi_0_rel, self._phi_rf)
            return self._phi_0_abs

        logging.error("The phase was not initialized. Returning None...")
        return None

    @phi_0_abs.deleter
    def phi_0_abs(self) -> None:
        """Delete attribute."""
        self._phi_0_abs = None

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

        if self._phi_0_abs is not None:
            if not hasattr(self, "phi_rf"):
                logging.error(
                    f"{self = }: cannot compute phi_0_rel from phi_0_abs if "
                    "phi_rf is not defined. Returning None..."
                )
                return None
            self.phi_0_rel = phi_0_abs_to_rel(self._phi_0_abs, self._phi_rf)
            return self._phi_0_rel

        if not hasattr(self, "_phi_s"):
            logging.error(
                f"{self = }: phi_0_abs, phi_0_rel, phi_s are all uninitialized"
                ". Returning None..."
            )
            return None

        phi_s_to_phi_0_rel = getattr(self, "_phi_s_to_phi_0_rel", None)
        if phi_s_to_phi_0_rel is None:
            logging.error(
                f"{self = }: you must set a function to compute phi_0_rel from"
                " phi_s with CavitySettings.set_cavity_parameters_methods"
                " method."
            )
            return None

        self.phi_0_rel = phi_s_to_phi_0_rel(self._phi_s)
        return self._phi_0_rel

    @phi_0_rel.deleter
    def phi_0_rel(self) -> None:
        """Delete attribute."""
        self._phi_0_rel = None

    # =============================================================================
    # Synchronous phase, accelerating voltage
    # =============================================================================
    @property
    def phi_s(self) -> None:
        """Get synchronous phase, compute it if necessary."""

    @phi_s.setter
    def phi_s(self, value: float) -> None:
        """Set the synchronous phase to desired value."""
        self._phi_s = value

    @phi_s.deleter
    def phi_s(self) -> None:
        """Delete the synchronous phase."""
        if not hasattr(self, "_phi_s"):
            return
        del self._phi_s

    @phi_s.getter
    def phi_s(self) -> float | None:
        """Get the synchronous phase, and compute it if necessary.

        .. note::
            It is mandatory for the calculation of this quantity to compute
            propagation of the particle in the cavity.

        See Also
        --------
        set_cavity_parameters_methods

        """
        if hasattr(self, "_phi_s"):
            return self._phi_s

        if not hasattr(self, "_phi_rf"):
            return None

        # We omit the _ in front of phi_0_rel to compute it if necessary
        if self.phi_0_rel is None:
            logging.error(
                "You must declare the particle entry phase in the "
                "cavity to compute phi_0_rel and then phi_s."
            )
            return None

        phi_s_calc = getattr(self, "_phi_0_rel_to_cavity_parameters", None)
        if phi_s_calc is None:
            logging.error(
                "You must set a function to compute phi_s from phi_0_rel with "
                "CavitySettings.set_cavity_parameters_arguments()"
            )
            return None

        self._phi_s = phi_s_calc(self.phi_0_rel)
        return self._phi_s

    @phi_s.deleter
    def phi_s(self) -> None:
        """Delete attribute."""
        self._phi_s = np.NaN

    def set_cavity_parameters_methods(
        self,
        solver_id: str,
        transf_mat_function_wrapper: Callable,
        phi_s_func: Callable | None = None,
    ) -> None:
        """Set the generic methods to compute beam propagation, cavity params.

        This function is called within two contexts:
            - When initializing the :class:`.BeamCalculator` specific
            parameters (:class:`.ElementBeamCalculatorParameters`).
            - When re-initalizing the :class:`.ElementBeamCalculatorParameters`
            because the ``status`` of the cavity changed, and in particular
            when it switches to ``'failed'``. In this case, the ``phi_s_func``
            is not altered.

        Parameters
        ----------
        solver_id : str
            The name of the solver for which functions must be changed.
        transf_mat_function_wrapper : Callable
            A function that compute the propagation of the beam.
        phi_s_func : Callable | None, optional
            A function that takes in the ouptut of
            ``transf_mat_function_wrapper`` and return the accelerating voltage
            in MV and the synchronous phase in rad. The default is None, which
            happens when we break the cavity and only the
            ``transf_mat_function_wrapper`` needs to be updated. In this case,
            the synchronous phase function is left unchanged.

        See Also
        --------
        set_cavity_parameters_arguments

        """
        self.transf_mat_func_wrappers[solver_id] = transf_mat_function_wrapper
        if phi_s_func is None:
            return
        self.phi_s_funcs[solver_id] = phi_s_func

    def set_cavity_parameters_arguments(
        self, solver_id: str, w_kin: float, **kwargs
    ) -> None:
        r"""Adapt the cavity parameters methods to beam with ``w_kin``.

        This function must be called every time the kinetic energy at the
        entrance of the cavity is changed (like this occurs during optimisation
        process) or when the synchronous phase must be calculated with another
        solver.

        Parameters
        ----------
        solver_id : str
            Name of the solver that will compute :math:`V_\mathrm{cav}` and
            :math:`\phi_s`.
        w_kin : float
            Kinetic energy of the synchronous particle at the entry of the
            cavity.
        kwargs :
            Other keyword arguments that will be passed to the function that
            will compute propagation of the beam in the :class:`.FieldMap`.
            Note that you should check that ``phi_0_rel`` key should be removed
            in your :class:`.BeamCalculator`, to avoid a clash in the
            :func:`phi_0_rel_to_cavity_parameters`.

        See Also
        --------
        set_cavity_parameters_methods

        """
        transf_mat_function_wrapper = _get_valid_func(
            self, "transf_mat_func_wrappers", solver_id
        )
        phi_s_func = _get_valid_func(self, "phi_s_funcs", solver_id)

        def phi_0_rel_to_cavity_parameters(
            phi_0_rel: float,
        ) -> tuple[float, float]:
            """Compute propagation of the beam, deduce v_cav and phi_s."""
            results = transf_mat_function_wrapper(
                phi_0_rel=phi_0_rel, w_kin_in=w_kin, **kwargs
            )
            cavity_parameters = phi_s_func(**results)
            return cavity_parameters

        def _residue_func(phi_0_rel: float, phi_s: float) -> float:
            """Compute difference between given and calculated ``phi_s``."""
            calculated_phi_s = phi_0_rel_to_cavity_parameters(phi_0_rel)[1]
            residue = diff_angle(phi_s, calculated_phi_s)
            return residue**2

        def phi_s_to_phi_0_rel(phi_s: float) -> float:
            """Optimise to find ``phi_0_rel`` matching ``phi_s``."""
            out = minimize_scalar(
                _residue_func, bounds=(0.0, 2.0 * math.pi), args=(phi_s,)
            )
            if not out.success:
                logging.error("Synch phase not found")
            return out.x

        self._phi_0_rel_to_cavity_parameters = phi_0_rel_to_cavity_parameters
        self._phi_s_to_phi_0_rel = phi_s_to_phi_0_rel

    @property
    def v_cav_mv(self) -> None:
        """Get accelerating voltage, compute it if necessary."""

    @v_cav_mv.setter
    def v_cav_mv(self, value: float) -> None:
        """Set accelerating voltage to desired value."""
        self._v_cav_mv = value

    @v_cav_mv.getter
    def v_cav_mv(self) -> float | None:
        """Get the accelerating voltage, and compute it if necessary.

        .. note::
            It is mandatory for the calculation of this quantity to compute
            propagation of the particle in the cavity.

        See Also
        --------
        set_cavity_parameters_methods

        """
        if hasattr(self, "_v_cav_mv"):
            return self._v_cav_mv

        # We omit the _ in front of phi_0_rel to compute it if necessary
        if self.phi_0_rel is None:
            logging.error(
                "You must declare the particle entry phase in the "
                "cavity to compute phi_0_rel and then v_cav_mv."
            )
            return None

        v_cav_mv_calc = getattr(self, "_phi_0_rel_to_v_cav_mv", None)
        if v_cav_mv_calc is None:
            logging.debug(
                "You must set a function to compute v_cav_mv from "
                "phi_0_rel with CavitySettings.set_cavity_parameters_arguments"
                " method."
            )
            return None

        raise NotImplementedError()

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
        self._phi_bunch = self.rf_phase_to_bunch_phase(value)
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
        """Declare the synchronous particle entry phase in bunch."""

    @phi_bunch.setter
    def phi_bunch(self, value: float) -> None:
        """Convert bunch to rf frequency."""
        self._phi_bunch = value
        self._phi_rf = self.bunch_phase_to_rf_phase(value)
        self._delete_non_reference_phases()

    @phi_bunch.getter
    def phi_bunch(self) -> float:
        """Return the entry phase of the synchronous particle (bunch ref)."""
        return self._phi_bunch

    def shift_phi_bunch(
        self, delta_phi_bunch: float, check_positive: bool = False
    ) -> None:
        """Shift the synchronous particle entry phase by ``delta_phi_bunch``.

        This is mandatory when the reference phase is changed. In particular,
        it is the case when studying a sub-list of elements with
        :class:`.TraceWin`. With this solver, the entry phase in the first
        element of the sub-:class:`.ListOfElements` is always 0.0, even if is
        not the first element of the linac.

        Parameters
        ----------
        delta_phi_bunch : float
            Phase difference between the new first element of the linac and the
            previous first element of the linac.

        Examples
        --------
        >>> phi_in_1st_element = 0.
        >>> phi_in_20th_element = 55.
        >>> 25th_element: FieldMap
        >>> 25th_element.cavity_settings.shift_phi_bunch(
        >>> ... phi_in_20th_element - phi_in_1st_element
        >>> )  # now phi_0_abs and phi_0_rel are properly understood

        """
        self.phi_bunch = self._phi_bunch - delta_phi_bunch
        if not check_positive:
            return
        assert (
            self.phi_bunch >= 0.0
        ), "The phase of the synchronous particle should never be negative."

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


def _get_valid_func(obj: object, func_name: str, solver_id: str) -> Callable:
    """Get the function in ``func_name`` for ``solver_id``."""
    all_funcs = getattr(obj, func_name, None)
    assert isinstance(all_funcs, dict), (
        f"Attribute {func_name} of {object} should be a dict[str, Callable] "
        f"but is {all_funcs}. "
        "Check CavitySettings.set_cavity_parameters_methods and"
        "CavitySettings.set_cavity_parameters_arguments"
    )
    func = all_funcs.get(solver_id, None)
    assert isinstance(func, Callable), (
        f"No Callable {func_name} was found in {object} for {solver_id = }"
        "Check CavitySettings.set_cavity_parameters_methods and"
        "CavitySettings.set_cavity_parameters_arguments"
    )
    return func
