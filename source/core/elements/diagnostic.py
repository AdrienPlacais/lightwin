#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`Diagnostic`, useless but increments index."""

from core.elements.element import Element


class Diagnostic(Element):
    """A dummy object."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 name: str | None = None,
                 **kwargs: str) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.
        self.idx['increment_lattice_idx'] = False
        self.idx['increment_elt_idx'] = True
