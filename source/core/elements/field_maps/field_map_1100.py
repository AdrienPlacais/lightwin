#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a field map with 1D rf electro-magnetic field."""
from core.elements.field_maps.field_map import FieldMap


class FieldMap1100(FieldMap):
    """1D rf electro-magnetic field.

    Just inherit from the classic :class:`.FieldMap`; no difference with
    :class:`.FieldMap100`.

    """

    def __init__(self, *args, **kwargs) -> None:
        """Init the same object as :class:`.FieldMap100`."""
        return super().__init__(*args, **kwargs)
