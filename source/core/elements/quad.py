#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 10:26:19 2021.

@author: placais

This module holds :class:`Quad`.

"""
from core.elements.element import Element


class Quad(Element):
    """Sub-class of Element, with parameters specific to QUADs."""

    def __init__(self, elem: list[str]) -> None:
        n_attributes = len(elem) - 1
        assert n_attributes in range(3, 10)
        super().__init__(elem)
        self.grad = float(elem[2])
