#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds :class:`Drift`.

"""

from core.elements.element import Element


class Drift(Element):
    """A simple drift tube."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Check that number of attributes is valid."""
        n_attributes = len(line) - 1
        assert n_attributes in [2, 3, 5]
        super().__init__(line, dat_idx)
