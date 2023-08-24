#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds :class`Solenoid`.

"""

from core.elements.element import Element


class Solenoid(Element):
    """Sub-class of Element, with parameters specific to SOLENOIDs."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes == 3
        super().__init__(elem)
