#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a :class:`SuperposedFieldMap`.

.. note::
    The initialisation of this class is particular, as it does not correspond
    to a specific line of the ``.dat`` file.

.. todo::
    Maybe move this to the field_map package once it is implemented

"""
from core.elements.element import Element


class SuperposedFieldMap(Element):
    """A single element holding several field maps."""

    is_implemented = True
    n_attributes = -1  # TODO what should it be?

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        total_length: float | None = None,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Save length of the superposed field maps."""
        super().__init__(line, dat_idx, name)
        self.length_m = total_length
