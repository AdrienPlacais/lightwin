#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.Element` objects."""
from typing import Any
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
    'FIELD_MAP': FieldMap,
    'QUAD': Quad,
    'SOLENOID': Solenoid,
    'THIN_STEERING': ThinSteering,
}  #:


class ElementFactory:
    """An object to create :class:`.Element` objects."""

    def __init__(self,
                 default_field_map_folder: str,
                 **factory_kw: Any) -> None:
        """Save the default folder for field maps."""
        self.default_field_map_folder = default_field_map_folder

    def run(self,
            line: list[str],
            dat_idx: int,
            **kwargs) -> Element:
        """Call proper constructor."""
        elt_name, line = self._personalized_name(line)
        element_creator = IMPLEMENTED_ELEMENTS[line[0]]
        element = element_creator(
            line,
            dat_idx,
            elt_name,
            default_field_map_folder=self.default_field_map_folder,
            **kwargs)
        return element

    def _personalized_name(self,
                           line: list[str]) -> tuple[str | None, list[str]]:
        """
        Extract the user-defined name of the Element if there is one.

        .. todo::
            Make this robust.

        """
        original_line = ' '.join(line)
        line_delimited_with_name = original_line.split(':', maxsplit=1)

        if len(line_delimited_with_name) == 2:
            elt_name = line_delimited_with_name[0]
            cleaned_line = line_delimited_with_name[1].split()
            return elt_name, cleaned_line

        return None, line
