#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`Drift`."""

from core.elements.element import Element


class Drift(Element):
    """A simple drift tube."""

    _id = "DR"

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Check that number of attributes is valid."""
        super().__init__(line, dat_idx, name)
        n_attributes = len(line) - 1
        assert n_attributes in (2, 3, 5)
