#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:33:26 2023.

@author: placais

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
