#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds :class:`Quad`."""
from core.elements.element import Element


class Quad(Element):
    """A partially defined quadrupole."""

    def __init__(self, line: list[str], dat_idx: int, **kwargs: str) -> None:
        """Check number of attributes, set gradient."""
        super().__init__(line, dat_idx)
        n_attributes = len(line) - 1
        assert n_attributes in range(3, 10)
        self.grad = float(line[2])

    @property
    def is_accelerating(self) -> bool:
        """Return ``False``."""
        return False
