#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`Solenoid`."""

from core.elements.element import Element


class Solenoid(Element):
    """A partially defined solenoid."""

    base_name = "SOL"

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Check number of attributes."""
        super().__init__(line, dat_idx, name)
        n_attributes = len(line) - 1
        assert n_attributes == 3
