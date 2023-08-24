#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds `Drift`.

"""

from core.elements.element import Element


class Drift(Element):
    """A simple drift tube."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes in [2, 3, 5]
        super().__init__(elem)
