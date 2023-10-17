#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Here we define a class to hold solver parameters for :class:`.Envelope1D`.

This module holds :class:`SingleElementEnvelope1DParameters`, that inherits
from the Abstract Base Class :class:`.SingleElementCalculatorParameters`.
It holds the transfer matrix function that is used, according to the solver
(Runge-Kutta or leapfrog) and their version (Python or Cython), as well as the
meshing in accelerating elements.

"""
from types import ModuleType

import numpy as np

import util.converters as convert
from core.electric_field import compute_param_cav
from beam_calculation.single_element_beam_calculator_parameters import (
    SingleElementCalculatorParameters)


class SingleElementEnvelope1DParameters(SingleElementCalculatorParameters):
    """
    Holds the parameters to compute beam propagation in an Element.

    has and get method inherited from SingleElementCalculatorParameters parent
    class.

    """

    def __init__(self, length_m: float, is_accelerating: bool,
                 n_cells: int | None, n_steps_per_cell: int, method: str,
                 transf_mat_module: ModuleType) -> None:
        """Set the actually useful parameters."""
        self.n_steps = 1

        self.n_cells = n_cells
        self.back_up_function = transf_mat_module.z_drift
        self.transf_mat_function = transf_mat_module.z_drift

        if is_accelerating:
            assert n_cells is not None
            self.n_steps = n_cells * n_steps_per_cell

            if method == 'RK':
                self.transf_mat_function = transf_mat_module.z_field_map_rk4
            elif method == 'leapfrog':
                self.transf_mat_function = \
                    transf_mat_module.z_field_map_leapfrog

        self.d_z = length_m / self.n_steps
        self.rel_mesh = np.linspace(0., length_m, self.n_steps + 1)

        self.s_in: int
        self.s_out: int
        self.abs_mesh: np.ndarray

    def set_absolute_meshes(self, pos_in: float, s_in: int
                            ) -> tuple[float, int]:
        """Set the absolute indexes and arrays, depending on previous elem."""
        self.abs_mesh = self.rel_mesh + pos_in

        self.s_in = s_in
        self.s_out = self.s_in + self.n_steps

        return self.abs_mesh[-1], self.s_out

    def re_set_for_broken_cavity(self):
        """Change solver parameters for efficiency purposes."""
        self.transf_mat_function = self.back_up_function

    # FIXME should not have dependencies is_accelerating, status
    def transf_mat_function_wrapper(self, w_kin_in: float,
                                    is_accelerating: bool, elt_status: str,
                                    **rf_field_kwargs) -> dict:
        """
        Calculate beam propagation in the Element.

        This wrapping is not very Pythonic, should be removed in the future.
        """
        gamma = convert.energy(w_kin_in, "kin to gamma")

        args = (self.d_z, gamma, self.n_steps)

        r_zz, gamma_phi, itg_field = self.transf_mat_function(
            *args, **rf_field_kwargs)

        cav_params = None
        if is_accelerating:
            gamma_phi[:, 1] /= self.n_cells
            cav_params = compute_param_cav(itg_field, elt_status)

        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        results = {'r_zz': r_zz, 'cav_params': cav_params,
                   'w_kin': w_kin, 'phi_rel': gamma_phi[:, 1]}

        return results
