#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`DummyElement`. It does nothing."""

from core.elements.element import Element


class DummyElement(Element):
    """A dummy object."""

    is_implemented = False

    def __init__(
        self,
        line: list[str],
        dat_idx: int,
        name: str | None = None,
        **kwargs: str,
    ) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, name)
        self.length_m = 0.0
