#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds :class:`Drift`."""

from core.elements.element import Element


class Drift(Element):
    """A simple drift tube."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 elt_name: str | None = None,
                 **kwargs: str) -> None:
        """Check that number of attributes is valid."""
        super().__init__(line, dat_idx, elt_name)
        n_attributes = len(line) - 1
        assert n_attributes in [2, 3, 5]
