#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds :class:`DummyElement`. It does nothing."""

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
        self.idx['increment_lattice_idx'] = False
        self.idx['increment_elt_idx'] = True

    @property
    def is_accelerating(self) -> bool:
        """Return ``False``."""
        return False
