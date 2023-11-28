#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a factory to create the solver parameters for every solver."""
from abc import ABC, ABCMeta, abstractmethod

from beam_calculation.parameters.element_parameters import (
    ElementBeamCalculatorParameters
)
from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap


class ElementBeamCalculatorParametersFactory(ABC):
    """Define a method to easily create the solver parameters."""

    @abstractmethod
    def run(self, elt: Element) -> ElementBeamCalculatorParameters:
        """Create the proper subclass of solver parameters, instantiate it."""
        pass

    @abstractmethod
    def _parameters_subclass(self, elt: Element) -> ABCMeta:
        """Select the parameters adapted to ``elt``."""
        pass
