#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a class to hold solver parameters for :class:`.Envelope1D`.

This module holds :class:`ElementEnvelope1DParameters`, that inherits
from the Abstract Base Class :class:`.ElementBeamCalculatorParameters`.
It holds the transfer matrix function that is used, according to the solver
(Runge-Kutta or leapfrog) and their version (Python or Cython), as well as the
meshing in accelerating elements.

.. todo::
    change how rf_fields is handled. not very clean but it works

"""
from abc import abstractmethod
from types import ModuleType
from typing import Any, Callable, Sequence

import numpy as np

from beam_calculation.parameters.element_parameters import (
    ElementBeamCalculatorParameters)
from core.elements.bend import Bend
from core.elements.element import Element
from core.electric_field import compute_param_cav
from core.elements.field_maps.field_map import FieldMap
import util.converters as convert


FIELD_MAP_INTEGRATION_METHOD_TO_FUNC = {
    "RK": lambda module: module.z_field_map_rk4,
    "leapfrog": lambda module: module.z_field_map_leapfrog,
}


class ElementEnvelope1DParameters(ElementBeamCalculatorParameters):
    """
    Holds the parameters to compute beam propagation in an Element.

    has and get method inherited from ElementBeamCalculatorParameters parent
    class.

    """

    def __init__(self,
                 transf_mat_function: Callable,
                 length_m: float,
                 n_steps: int = 1,
                 ) -> None:
        """Set the actually useful parameters."""
        self.transf_mat_function = transf_mat_function

        self.n_steps = n_steps
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
        raise IOError("Calling this method for a non-field map is incorrect.")

    @abstractmethod
    def transfer_matrix_arguments(self) -> Sequence[Any]:
        """Give the element parameters necessary to compute transfer matrix."""

    def transf_mat_function_wrapper(self,
                                    w_kin_in: float,
                                    **rf_field_kwargs) -> dict:
        """Calculate beam propagation in the :class:`.Element`."""
        gamma_in = convert.energy(w_kin_in, "kin to gamma")
        r_zz, gamma_phi, itg_field = self.transf_mat_function(
            gamma_in,
            *self.transfer_matrix_arguments(),
            **rf_field_kwargs)

        results = self._transfer_matrix_results_to_dict(r_zz,
                                                        gamma_phi,
                                                        itg_field)
        return results

    def _transfer_matrix_results_to_dict(self,
                                         r_zz: np.ndarray,
                                         gamma_phi: np.ndarray,
                                         itg_field: float | None,
                                         ) -> dict:
        """Convert the results given by the transf_mat function to dict."""
        assert itg_field is None
        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        results = {'r_zz': r_zz,
                   'cav_params': None,
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results

    def _transfer_matrix_results_to_dict_broken_field_map(
            self,
            r_zz: np.ndarray,
            gamma_phi: np.ndarray,
            itg_field: float | None,
            ) -> dict:
        """Convert the results given by the transf_mat function to dict.

        This method should override the default
        ``_transfer_matrix_results_to_dict`` when the element under study is a
        broken field map.

        """
        assert itg_field is None
        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        results = {'r_zz': r_zz,
                   'cav_params': {'v_cav_mv': np.NaN, 'phi_s': np.NaN},
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results


class DriftEnvelope1DParameters(ElementEnvelope1DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.Drift`.

    As this is 1D, it is also used for :class:`.Solenoid`, :class:`.Quad`,
    broken :class:`.FieldMap`.

    """

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: Element,
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        transf_mat_function = transf_mat_module.z_drift
        super().__init__(
            transf_mat_function,
            elt.length_m,
        )

    def transfer_matrix_arguments(self) -> tuple[float, int]:
        """Give the element parameters necessary to compute transfer matrix."""
        return self.d_z, self.n_steps


class FieldMapEnvelope1DParameters(ElementEnvelope1DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.FieldMap`.

    Non-accelerating cavities will use :class:`.DriftEnvelope1DParameters`
    instead.

    """

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: FieldMap,
                 n_steps: int,
                 method: str,
                 n_steps_per_cell: int,
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        transf_mat_function = FIELD_MAP_INTEGRATION_METHOD_TO_FUNC[method](
            transf_mat_module)

        self.n_cell = int(elt.get('n_cell'))
        n_steps = self.n_cell * n_steps_per_cell
        super().__init__(transf_mat_function,
                         elt.length_m,
                         n_steps,
                         )
        self._transf_mat_module = transf_mat_module

    def transfer_matrix_arguments(self
                                  ) -> tuple[float, int]:
        """Give the element parameters necessary to compute transfer matrix."""
        return self.d_z, self.n_steps

    def _transfer_matrix_results_to_dict(self,
                                         r_zz: np.ndarray,
                                         gamma_phi: np.ndarray,
                                         itg_field: float | None,
                                         ) -> dict:
        """
        Convert the results given by the transf_mat function to dict.

        Overrides the default method defined in the ABC.

        """
        assert itg_field is not None
        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        gamma_phi[:, 1] /= self.n_cell
        cav_params = compute_param_cav(itg_field)
        results = {'r_zz': r_zz,
                   'cav_params': cav_params,
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results

    def re_set_for_broken_cavity(self) -> None:
        """Make beam calculator call Drift func instead of FieldMap."""
        self.transf_mat_function = self._transf_mat_module.z_drift

        def _new_transfer_matrix_results_to_dict(r_zz: np.ndarray,
                                                 gamma_phi: np.ndarray,
                                                 itg_field: float | None,
                                                 ) -> dict:
            """
            Convert the results given by the transf_mat function to dict.

            Overrides the default method defined in the ABC.

            """
            assert itg_field is None
            w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
            cav_params = compute_param_cav(np.NaN)
            results = {'r_zz': r_zz,
                       'cav_params': cav_params,
                       'w_kin': w_kin,
                       'phi_rel': gamma_phi[:, 1]
                       }
            return results

        self._transfer_matrix_results_to_dict = \
            _new_transfer_matrix_results_to_dict


class BendEnvelope1DParameters(ElementEnvelope1DParameters):
    """Hold the specific parameters to compute :class:`.Bend` transfer matrix.

    In particular, we define ``factor_1``, ``factor_2`` and ``factor_3`` to
    speed-up calculations.

    """

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: Bend,
                 **kwargs: str):
        """Instantiate object and pre-compute some parameters for speed.

        Parameters
        ----------
        transf_mat_module : ModuleType
            Module where the transfer matrix function is defined.
        elt : Bend
            ``BEND`` element.
        kwargs :
            kwargs

        """
        assert hasattr(transf_mat_module, "z_bend"), "BEND transfer matrix " \
            "not implement yet in Cython."
        transf_mat_function = transf_mat_module.z_bend

        super().__init__(transf_mat_function,
                         elt.length_m,
                         n_steps=1,
                         )

        h_squared = (elt.bend_angle
                     / abs(elt.curvature_radius * elt.bend_angle))**2
        k_x = np.sqrt((1. - elt.field_grad_index) * h_squared)
        factors = self._pre_compute_factors_for_transfer_matrix(elt.length_m,
                                                                h_squared,
                                                                k_x)
        self.factor_1, self.factor_2, self.factor_3 = factors

    def _pre_compute_factors_for_transfer_matrix(
        self,
        length_m: float,
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
        factor_1 = -h_squared * length_m / k_x**2
        factor_2 = h_squared * np.sin(k_x * length_m) / k_x**3
        factor_3 = length_m * (1. - h_squared / k_x**2)
        assert isinstance(factor_1, float)
        assert isinstance(factor_2, float)
        assert isinstance(factor_3, float)
        return factor_1, factor_2, factor_3

    def transfer_matrix_arguments(self) -> tuple[float, float, float, float]:
        """Give the element parameters necessary to compute transfer matrix."""
        return (self.d_z, self.factor_1, self.factor_2, self.factor_3)
