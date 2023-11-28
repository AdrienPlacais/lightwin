#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the solver parameters for :class:`.Envelope3D`."""
from abc import ABCMeta

from beam_calculation.parameters.factory import (
    ElementBeamCalculatorParametersFactory)
from beam_calculation.envelope_3d.\
    element_envelope3d_parameters import (
        BendEnvelope3DParameters,
        DriftEnvelope3DParameters,
        ElementEnvelope3DParameters,
        FieldMapEnvelope3DParameters,
        QuadEnvelope3DParameters,
        SolenoidEnvelope3DParameters,
    )
from core.elements.element import Element
from core.elements.bend import Bend
from core.elements.drift import Drift
from core.elements.field_maps.field_map import FieldMap
from core.elements.field_maps.field_map_100 import FieldMap100
from core.elements.field_maps.field_map_7700 import FieldMap7700
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid


IMPLEMENTED_ELEMENTS = {
    Bend: BendEnvelope3DParameters,
    Drift: DriftEnvelope3DParameters,
    FieldMap100: FieldMapEnvelope3DParameters,
    FieldMap7700: FieldMapEnvelope3DParameters,
    Quad: QuadEnvelope3DParameters,
    Solenoid: SolenoidEnvelope3DParameters,
}  #:


class ElementEnvelope3DParametersFactory(
        ElementBeamCalculatorParametersFactory):
    """Define a method to easily create the solver parameters."""

    def __init__(self,
                 method: str,
                 n_steps_per_cell: int,
                 flag_cython: bool = False):
        """Prepare import of proper functions."""
        assert method in ('leapfrog', 'RK')
        self.method = method
        self.n_steps_per_cell = n_steps_per_cell

        if flag_cython:
            raise NotImplementedError
        import beam_calculation.envelope_3d.transfer_matrices_p as \
            transf_mat_module
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
            'method': self.method,
            'n_steps_per_cell': self.n_steps_per_cell
        }
        subclass = self._parameters_subclass(elt)

        single_element_envelope_1d_parameters = subclass(
            self.transf_mat_module,
            elt,
            n_steps=1,
            **kwargs)

        return single_element_envelope_1d_parameters

    def _parameters_subclass(self, elt: Element) -> ABCMeta:
        """Select the parameters adapted to ``elt``.

        In particular, if the element is a non-accelerating
        :class:`.FieldMap`, we return the same parameters class as
        :class:`.Drift`.

        Parameters
        ----------
        elt : Element
            Element under study.

        Returns
        -------
        ABCMeta
            Proper subclass of :class:`.ElementEnvelope3DParameters`, not
            instantiated yet.

        """
        element_subclass = type(elt)
        assert element_subclass in IMPLEMENTED_ELEMENTS, "Solver parameters "\
            f"not implemented for {elt = } of type {element_subclass}."
        element_parameters_class = IMPLEMENTED_ELEMENTS[element_subclass]

        if not isinstance(elt, FieldMap):
            return element_parameters_class

        if elt.is_accelerating:
            return element_parameters_class

        return IMPLEMENTED_ELEMENTS[Drift]
