#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds :class:`ThinSteering`. It does nothing.

.. todo::
    Should it be ignored by lattice?

"""

from core.elements.element import Element


class ThinSteering(Element):
    """A dummy object."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 name: str | None = None,
                 **kwargs: str) -> None:
        """Force an element with null-length."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.
