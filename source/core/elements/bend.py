#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module holds :class:`Bend`. It just holds its length."""
import numpy as np

from core.elements.element import Element


class Bend(Element):
    """A dummy object."""

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 elt_name: str | None = None,
                 **kwargs: str) -> None:
        """Force an element with null-length, with no index."""
        super().__init__(line, dat_idx, elt_name)
        bend_angle = np.deg2rad(float(line[1]))
        curvature_radius = float(line[2])
        self.length_m = curvature_radius * np.abs(bend_angle) * 1e-3
        self.idx['increment_lattice_idx'] = False
        self.idx['increment_elt_idx'] = True
