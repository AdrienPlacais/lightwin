#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds :class:`SuperposedFieldMap`.

.. note::
    The initialisation of this class is particular, as it does not correspond
    to a specific line of the ``.dat`` file.

"""
from core.elements.element import Element
from core.elements.field_map import FieldMap


class SuperposedFieldMap(Element):
    """A single element holding several field maps."""

    def __init__(self,
                 line: str,
                 dat_idx: int,
                 total_length: float | None = None,
                 **kwargs: str) -> None:
        super().__init__(line, dat_idx)
        self.length_m = total_length

    @property
    def is_accelerating(self) -> bool:
        """Return ``False``."""
        return False
