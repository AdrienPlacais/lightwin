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
        bend_angle = float(np.deg2rad(float(line[1])))
        curvature_radius = float(line[2]) * 1e-3
        field_grad_index = int(line[3])
        self.length_m = curvature_radius * abs(bend_angle)

        # For transfer matrix -> to move elsewhere? Not used by TW
        h_squared = (bend_angle / abs(curvature_radius * bend_angle))**2
        k_x = np.sqrt((1. - field_grad_index) * h_squared)
        self.factors = self._pre_compute_factors_for_transfer_matrix(h_squared,
                                                                     k_x)

        self.idx['increment_lattice_idx'] = False
        self.idx['increment_elt_idx'] = True

    def _pre_compute_factors_for_transfer_matrix(
        self,
        h_squared: float,
        k_x: float
    ) -> tuple[float, float, float]:
        r"""
        Compute factors to speed up the transfer matrix calculation.

        ``factor_1`` is:

        .. math::
            \frac{-h^2\Delta s}{k_x^2}

        ``factor_2`` is:

        .. math::
            \frac{h^2 \sin{(k_x\Delta s)}}{k_x^3}

        ``factor_3`` is:

        .. math::
            \Delta s \left(1 - \frac{h^2}{k_x^2}\right)

        """
        factor_1 = -h_squared * self.length_m / k_x**2
        factor_2 = h_squared * np.sin(k_x * self.length_m) / k_x**3
        factor_3 = self.length_m * (1. - h_squared / k_x**2)
        assert isinstance(factor_1, float)
        assert isinstance(factor_2, float)
        assert isinstance(factor_3, float)
        return factor_1, factor_2, factor_3
