#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the solver parameters for :class:`.Envelope3D`."""
import logging

from beam_calculation.envelope_3d.element_envelope3d_parameters import (
    BendEnvelope3DParameters,
    DriftEnvelope3DParameters,
    ElementEnvelope3DParameters,
    FieldMapEnvelope3DParameters,
    QuadEnvelope3DParameters,
    SolenoidEnvelope3DParameters,
)
from beam_calculation.parameters.factory import (
    ElementBeamCalculatorParametersFactory,
)
from core.elements.bend import Bend
from core.elements.diagnostic import Diagnostic
from core.elements.drift import Drift
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid

PARAMETERS_3D = {
    Bend: BendEnvelope3DParameters,
    Diagnostic: DriftEnvelope3DParameters,
    Drift: DriftEnvelope3DParameters,
    FieldMap: FieldMapEnvelope3DParameters,
    Quad: QuadEnvelope3DParameters,
    Solenoid: SolenoidEnvelope3DParameters,
}  #:


class ElementEnvelope3DParametersFactory(
    ElementBeamCalculatorParametersFactory
):
    """Define a method to easily create the solver parameters."""

    def __init__(
        self,
        method: str,
        n_steps_per_cell: int,
        solver_id: str,
        flag_cython: bool = False,
        phi_s_definition: str = "historical",
    ) -> None:
        """Prepare import of proper functions."""
        assert method in ("RK", "RK4")
        self.method = method
        self.n_steps_per_cell = n_steps_per_cell
        self.solver_id = solver_id
        self.phi_s_definition = phi_s_definition

        if flag_cython:
            raise NotImplementedError
        import beam_calculation.envelope_3d.transfer_matrices_p as transf_mat_module

        self.transf_mat_module = transf_mat_module

    def run(self, elt: Element) -> ElementEnvelope3DParameters:
        """Create the proper subclass of solver parameters, instantiate it.

        Parameters
        ----------
        elt : Element
            Element under study.

        Returns
        -------
        ElementEnvelope3DParameters
            Proper instantiated subclass of
            :class:`.ElementEnvelope3DParameters`.

        """
        kwargs = {
            "method": self.method,
            "n_steps_per_cell": self.n_steps_per_cell,
            "solver_id": self.solver_id,
            "phi_s_definition": self.phi_s_definition,
        }
        subclass = self._parameters_constructor(elt)

        single_element_envelope_3d_parameters = subclass(
            self.transf_mat_module, elt, **kwargs
        )

        return single_element_envelope_3d_parameters

    def _parameters_constructor(
        self, elt: Element, default: type = PARAMETERS_3D[Drift]
    ) -> type:
        """Get the proper object constructor.

        Examples
        --------
        >>> self._parameters_constructor(Drift())
        DriftEnvelope3DParameters

        >>> self._parameters_constructor(Quad())
        QuadEnvelope3DParameters

        As DiagPosition is not in PARAMETERS_3D, we look for the mother
        class Diagnostic.

        >>> self._parameters_constructor(DiagPosition())
        DriftEnvelope3DParameters

        To avoid wasting computation time, non-accelerating field maps are
        treated as drifts.

        >>> self._parameters_constructor(FieldMap100(is_accelerating=False))
        DriftEnvelope3DParameters

        """
        if isinstance(elt, FieldMap) and not elt.is_accelerating:
            return default

        element_class = type(elt)
        constructor = PARAMETERS_3D.get(element_class, None)
        if constructor is not None:
            return constructor

        super_class = element_class.__base__
        constructor = PARAMETERS_3D.get(super_class, None)
        if constructor is not None:
            return constructor

        logging.error(
            f"Element {elt} of {element_class = } not added to the Envelope3D "
            "dict linking every Element class to its specific parameters"
            "(transfer matrix in particular). Neither was found its "
            f"{super_class = }. "
            "Note that you can use the elements_to_remove key in the "
            "Envelope3D.ListOfElementFactory class."
        )
        return default
