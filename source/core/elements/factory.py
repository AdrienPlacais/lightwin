#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.Element` objects."""
from core.elements.element import Element
from core.elements.aperture import Aperture
from core.elements.drift import Drift
from core.elements.dummy import DummyElement
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid
from core.elements.thin_steering import ThinSteering

IMPLEMENTED_ELEMENTS = {
    'APERTURE': Aperture,
    'DRIFT': Drift,
    'DUMMY_ELEMENT': DummyElement,
    'FIELDMAP': FieldMap,
    'QUAD': Quad,
    'SOLENOID': Solenoid,
    'THIN_STEERING': ThinSteering,
}  #:


class ElementFactory:
    """An object to create :class:`.Element` objects."""

    def __init__(self) -> None:
        """Do nothing for now.

        .. todo::
            Check if it would be relatable to hold some arguments? As for now,
            I would be better off with a run function instead of a class.

        """
        return

    def run(self,
            line: list[str],
            dat_idx: int,
            **kwargs) -> Element:
        """Call proper constructor."""
        elt_name, line = self._personalized_name(line)
        element = Element(line, dat_idx, elt_name, **kwargs)
        return element

    def _personalized_name(self,
                           line: list[str]) -> tuple[str | None, list[str]]:
        """
        Extract the user-defined name of the Element if there is one.

        .. todo::
            Make this robust.

        """
        if line[1] == ':':
            elt_name = line[0]
            del line[1]
            del line[0]
            return elt_name, line
        return None, line
