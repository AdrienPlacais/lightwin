#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define an horizontal :class:`Bend`."""
import numpy as np

from core.elements.element import Element


class Bend(Element):
    """This class holds the necessary to compute longitudinal transfer matrix.

    In TraceWin documentation, transfer matrix is in :math:`mm` and
    :math:`deg`. We use here :math:`m` and :math:`rad`.

    """

    def __init__(self,
                 line: list[str],
                 dat_idx: int,
                 elt_name: str | None = None,
                 **kwargs: str) -> None:
        """Precompute the parameters used to compute transfer matrix."""
        super().__init__(line, dat_idx, elt_name)

        # Basics
        self.bend_angle = float(np.deg2rad(float(line[1])))
        self.curvature_radius = float(line[2]) * 1e-3
        self.field_grad_index = float(line[3])
        self.length_m = self.curvature_radius * abs(self.bend_angle)

        self.idx['increment_lattice_idx'] = False
        self.idx['increment_elt_idx'] = True
