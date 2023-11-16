#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module defines a field map with 1D rf electric field."""
from core.elements.field_maps.field_map import FieldMap


class FieldMap100(FieldMap):
    """1D rf electric field along ``z``.

    It corresponds to the legacy :class:`.FieldMap`. This is the only kind of
    ``FIELD_MAP`` currently supported by :class:`.Envelope1D` (although 1D
    static electric field could also be implemented).

    As for now, it only inherits from the :class:`.FieldMap` and does not bring
    anything new.

    """
