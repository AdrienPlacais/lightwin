#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds :class:`Aperture`. It does nothing.

.. todo::
    Should it be ignored by lattice?

"""

from core.elements.element import Element


class Aperture(Element):
    """A dummy object."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Force an element with null-length."""
        super().__init__(line, dat_idx)
        self.length_m = 0.

    @property
    def is_accelerating(self) -> bool:
        """Return ``False``."""
        return False
