#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Create the solver parameters for :class:`.Envelope1D`."""
from abc import ABCMeta
import logging

from beam_calculation.parameters.factory import (
    ElementBeamCalculatorParametersFactory)
from beam_calculation.envelope_1d.\
    element_envelope1d_parameters import (
        BendEnvelope1DParameters,
        DriftEnvelope1DParameters,
        ElementEnvelope1DParameters,
        FieldMapEnvelope1DParameters,
    )
from core.elements.aperture import Aperture
from core.elements.bend import Bend
from core.elements.diagnostic import Diagnostic
from core.elements.drift import Drift
from core.elements.edge import Edge
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap
from core.elements.field_maps.field_map_100 import FieldMap100
from core.elements.field_maps.field_map_1100 import FieldMap1100
from core.elements.field_maps.field_map_7700 import FieldMap7700
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid


IMPLEMENTED_ELEMENTS = {
    Aperture: DriftEnvelope1DParameters,
    Bend: BendEnvelope1DParameters,
    Edge: DriftEnvelope1DParameters,
    Diagnostic: DriftEnvelope1DParameters,
    Drift: DriftEnvelope1DParameters,
    FieldMap100: FieldMapEnvelope1DParameters,
    FieldMap1100: FieldMapEnvelope1DParameters,
    FieldMap7700: FieldMapEnvelope1DParameters,
    Quad: DriftEnvelope1DParameters,
    Solenoid: DriftEnvelope1DParameters,
}  #:


class ElementEnvelope1DParametersFactory(
        ElementBeamCalculatorParametersFactory):
    """Define a method to easily create the solver parameters."""

    def __init__(self,
                 method: str,
                 n_steps_per_cell: int,
                 solver_id: str,
                 flag_cython: bool = False,
                 phi_s_definition: str = 'historical') -> None:
        """Prepare import of proper functions."""
        assert method in ('leapfrog', 'RK', 'RK4')
        self.method = method
        self.n_steps_per_cell = n_steps_per_cell
        self.solver_id = solver_id
        self.phi_s_definition = phi_s_definition

        if flag_cython:
            try:
                import beam_calculation.envelope_1d.transfer_matrices_c as \
                    transf_mat_module
            except ModuleNotFoundError:
                logging.error("Cython not found. Maybe it was not compilated. "
                              "Check util/setup.py for information.")
                raise ModuleNotFoundError
        else:
            import beam_calculation.envelope_1d.transfer_matrices_p as \
                transf_mat_module
        self.transf_mat_module = transf_mat_module

    def run(self, elt: Element) -> ElementEnvelope1DParameters:
        """Create the proper subclass of solver parameters, instantiate it.

        Parameters
        ----------
        elt : Element
            Element under study.

        Returns
        -------
        ElementEnvelope1DParameters
            Proper instantiated subclass of
            :class:`.ElementEnvelope1DParameters`.

        """
        kwargs = {
            'method': self.method,
            'n_steps_per_cell': self.n_steps_per_cell,
            'solver_id': self.solver_id,
            'phi_s_definition': self.phi_s_definition,
        }
        subclass = self._parameters_subclass(elt)

        single_element_envelope_1d_parameters = subclass(
            self.transf_mat_module,
            elt=elt,
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
            Proper subclass of :class:`.ElementEnvelope1DParameters`, not
            instantiated yet.

        """
        element_subclass = type(elt)

        self._check_is_implemented(element_subclass)
        element_parameters_class = IMPLEMENTED_ELEMENTS[element_subclass]

        if not isinstance(elt, FieldMap):
            return element_parameters_class

        if elt.is_accelerating:
            return element_parameters_class

        return IMPLEMENTED_ELEMENTS[Drift]

    def _check_is_implemented(self, element_subclass: ABCMeta) -> None:
        """Check if the element under study has a defined transfer matrix."""
        if element_subclass not in IMPLEMENTED_ELEMENTS:
            logging.error(f"The element subclass {element_subclass} is not "
                          "implemented. Implement it, or use the "
                          "elements_to_remove key in the "
                          "BeamCalculator.ListOfElementFactory class.")
            raise IOError
