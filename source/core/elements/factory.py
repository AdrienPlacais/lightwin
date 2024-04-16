#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.Element` objects."""
from pathlib import Path
from typing import Any

from core.elements.aperture import Aperture
from core.elements.bend import Bend
from core.elements.diagnostic import Diagnostic
from core.elements.drift import Drift
from core.elements.dummy import DummyElement
from core.elements.edge import Edge
from core.elements.element import Element
from core.elements.field_maps.factory import FieldMapFactory
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid
from core.elements.thin_steering import ThinSteering

IMPLEMENTED_ELEMENTS = {
    "APERTURE": Aperture,
    "BEND": Bend,
    "DRIFT": Drift,
    "DIAG_ACHROMAT": Diagnostic,
    "DIAG_DENERGY": Diagnostic,
    "DIAG_DPHASE": Diagnostic,
    "DIAG_DSIZE": Diagnostic,
    "DIAG_DSIZE2": Diagnostic,
    "DIAG_DSIZE3": Diagnostic,
    "DIAG_DSIZE4": Diagnostic,
    "DIAG_ENERGY": Diagnostic,
    "DIAG_POSITION": Diagnostic,
    "DIAG_SIZE": Diagnostic,
    "DIAG_SIZE_MAX": Diagnostic,
    "DIAG_TWISS": Diagnostic,
    "DIAG_WAIST": Diagnostic,
    "DUMMY_ELEMENT": DummyElement,
    "EDGE": Edge,
    "FIELD_MAP": FieldMap,  # replaced in ElementFactory initialisation
    "QUAD": Quad,
    "SOLENOID": Solenoid,
    "THIN_STEERING": ThinSteering,
}  #:


class ElementFactory:
    """An object to create :class:`.Element` objects."""

    def __init__(
        self,
        default_field_map_folder: Path,
        freq_bunch_mhz: float,
        **factory_kw: Any,
    ) -> None:
        """Create a factory for the field maps."""
        field_map_factory = FieldMapFactory(
            default_field_map_folder, freq_bunch_mhz, **factory_kw
        )
        self.field_map_factory = field_map_factory
        IMPLEMENTED_ELEMENTS["FIELD_MAP"] = field_map_factory.run

    def run(self, line: list[str], dat_idx: int, **kwargs) -> Element:
        """Call proper constructor."""
        name, line = self._personalized_name(line)
        element_creator = IMPLEMENTED_ELEMENTS[line[0].upper()]
        element = element_creator(line, dat_idx, name, **kwargs)
        return element

    def _personalized_name(
        self, line: list[str]
    ) -> tuple[str | None, list[str]]:
        """
        Extract the user-defined name of the Element if there is one.

        .. todo::
            Make this robust.

        """
        original_line = " ".join(line)
        line_delimited_with_name = original_line.split(":", maxsplit=1)

        if len(line_delimited_with_name) == 2:
            name = line_delimited_with_name[0].strip()
            cleaned_line = line_delimited_with_name[1].split()
            return name, cleaned_line

        return None, line
