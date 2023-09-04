#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 20:27:52 2023.

@author: placais

This module holds :class:`DummyElement`. It does nothing.

.. todo::
    Should be ignored by lattice

"""

from core.elements.element import Element


class DummyElement(Element):
    """A dummy object."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 **kwargs: str) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx)
        self.length_m = 0.
        self.idx['increment_idx'] = False
