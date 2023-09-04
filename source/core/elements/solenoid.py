#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds :class:`Solenoid`.

"""

from core.elements.element import Element


class Solenoid(Element):
    """A partially defined solenoid."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Check number of attributes."""
        super().__init__(line, dat_idx)
        n_attributes = len(line) - 1
        assert n_attributes == 3
