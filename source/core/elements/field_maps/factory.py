#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily create :class:`.FieldMap` objects.

This element has it's own factory as I expect that creating field maps will
become very complex in the future: 3D, superposed fields...

"""
from typing import Any
from core.elements.field_maps.field_map import FieldMap


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

        field_map = FieldMap(
            line,
            dat_idx,
            elt_name=elt_name,
            default_field_map_folder=self.default_field_map_folder
        )
        return field_map

    def _append_absolute_phase_flag(self, line: list[str]) -> None:
        """Add an explicit absolute phase flag."""
        line.append(self.default_absolute_phase_flag)

