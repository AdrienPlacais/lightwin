#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.FieldMap` objects.

This element has it's own factory as I expect that creating field maps will
become very complex in the future: 3D, superposed fields...

.. todo::
    This will be subclassed, as the differennt solver do not have the same
    needs. :class:`.TraceWin` does not need to load the electromagnetic fields,
    so every ``FIELD_MAP`` is implemented.
    :class:`.Envelope1D` cannot support 3D.
    etc

"""
from typing import Any
from abc import ABCMeta
import logging

from core.elements.field_maps.field_map import FieldMap
from core.elements.field_maps.field_map_100 import FieldMap100
from core.elements.field_maps.field_map_7700 import FieldMap7700


IMPLEMENTED_FIELD_MAPS = {
    100: FieldMap100,
    7700: FieldMap7700,
    }  #:
IMPLEMENTATION_WARNING_ALREADY_RAISED = False


class FieldMapFactory:
    """An object to create :class:`.FieldMap` objects."""

    def __init__(self,
                 default_field_map_folder: str,
                 default_absolute_phase_flag: str = '0',
                 **factory_kw: Any) -> None:
        """Save the default folder for field maps."""
        self.default_field_map_folder = default_field_map_folder
        self.default_absolute_phase_flag = default_absolute_phase_flag

    def run(self,
            line: list[str],
            dat_idx: int,
            elt_name: str | None = None,
            **kwargs) -> FieldMap:
        """Call proper constructor."""
        if len(line) == 10:
            self._append_absolute_phase_flag(line)

        field_map_class = self._get_proper_field_map_subclass(int(line[1]))

        field_map = field_map_class(
            line,
            dat_idx,
            elt_name=elt_name,
            default_field_map_folder=self.default_field_map_folder
        )
        return field_map

    def _append_absolute_phase_flag(self, line: list[str]) -> None:
        """Add an explicit absolute phase flag."""
        line.append(self.default_absolute_phase_flag)

    def _get_proper_field_map_subclass(self, geometry: int) -> ABCMeta:
        """Determine the proper field map subclass.

        .. warning::
            As for now, it always raises an error or return the rf electric
            field 1D class :class:`.FieldMap100`.

        """
        if geometry not in IMPLEMENTED_FIELD_MAPS:
            raise NotImplementedError(f"{geometry = } not supported")

        if geometry == 7700:
            if not IMPLEMENTATION_WARNING_ALREADY_RAISED:
                logging.warning(
                    f"3D field maps ({geometry = }) not implemented "
                    "yet. If solver is Envelope1D or Envelope3D, "
                    "only the longitudinal rf electric field will be "
                    "used (equivalent of 'FIELD_MAP 100').")
                IMPLEMENTATION_WARNING_ALREADY_RAISED = True
            return FieldMap100

        return IMPLEMENTED_FIELD_MAPS[geometry]
