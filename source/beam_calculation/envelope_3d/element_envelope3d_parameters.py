#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define a class to hold solver parameters for :class:`.Envelope3D`.

This module holds :class:`ElementEnvelope3DParameters`, that inherits
from the Abstract Base Class :class:`.ElementCalculatorParameters`.
It holds the transfer matrix function that is used, as well as the meshing in
accelerating elements.

In a first time, only Runge-Kutta (no leapfrog) and only Python (no Cython).

The list of implemented transfer matrices is
:data:`implemented_transfer_matrices`.

"""
from abc import abstractmethod
from typing import Any, Callable, Sequence
from types import ModuleType

import numpy as np

from beam_calculation.envelope_1d.element_envelope1d_parameters import (
    ElementEnvelope1DParameters)
from core.electric_field import compute_param_cav
from core.elements.bend import Bend
from core.elements.drift import Drift
from core.elements.field_maps.field_map import FieldMap
from core.elements.quad import Quad
from core.elements.solenoid import Solenoid
import util.converters as convert
from util.synchronous_phases import SYNCHRONOUS_PHASE_FUNCTIONS


FIELD_MAP_INTEGRATION_METHOD_TO_FUNC = {
    'RK': lambda transf_mat_module: transf_mat_module.field_map_rk4,
    'RK4': lambda transf_mat_module: transf_mat_module.field_map_rk4,
}


class ElementEnvelope3DParameters(ElementEnvelope1DParameters):
    """Hold the parameters to compute beam propagation in an :class:`.Element`.

    has and get method inherited from ElementCalculatorParameters parent
    class.

    """

    def __init__(self,
                 transf_mat_function: Callable,
                 length_m: float,
                 n_steps: int = 1) -> None:
        """Save useful parameters as attribute.

        Parameters
        ----------
        transf_mat_function : Callable
            transf_mat_function
        length_m : float
            length_m
        n_steps : int
            n_steps

        """
        super().__init__(transf_mat_function, length_m, n_steps)

    @abstractmethod
    def transfer_matrix_arguments(self) -> Sequence[Any]:
        """Give the element parameters necessary to compute transfer matrix."""

    def _transfer_matrix_results_to_dict(self,
                                         transfer_matrix: np.ndarray,
                                         gamma_phi: np.ndarray,
                                         itg_field: float | None,
                                         ) -> dict:
        """Convert the results given by the transf_mat function to dict."""
        assert itg_field is None
        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        results = {'transfer_matrix': transfer_matrix,
                   'r_zz': transfer_matrix[:, 4:, 4:],
                   'cav_params': None,
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results

    def _transfer_matrix_results_to_dict_broken_field_map(
            self,
            transfer_matrix: np.ndarray,
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
        results = {'transfer_matrix': transfer_matrix,
                   'r_zz': transfer_matrix[:, 4:, 4:],
                   'cav_params': {'v_cav_mv': np.NaN, 'phi_s': np.NaN},
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results

    def re_set_for_broken_cavity(self):
        """Change solver parameters for efficiency purposes."""
        raise IOError("Calling this method for a non-field map is incorrect.")


class DriftEnvelope3DParameters(ElementEnvelope3DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.Drift`."""

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: Drift | FieldMap,
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        transf_mat_function = transf_mat_module.drift
        super().__init__(
            transf_mat_function,
            elt.length_m,
        )

    def transfer_matrix_arguments(self) -> tuple[float, int]:
        """Give the element parameters necessary to compute transfer matrix."""
        return self.d_z, self.n_steps


class QuadEnvelope3DParameters(ElementEnvelope3DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.Quad`."""

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: Quad,
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        transf_mat_function = transf_mat_module.quad
        super().__init__(
            transf_mat_function,
            elt.length_m,
        )
        self.gradient = elt.grad

    def transfer_matrix_arguments(self) -> tuple[float, float]:
        """Give the element parameters necessary to compute transfer matrix."""
        return self.d_z, self.gradient


class SolenoidEnvelope3DParameters(ElementEnvelope3DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.Quad`."""

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: Solenoid,
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        raise NotImplementedError


class FieldMapEnvelope3DParameters(ElementEnvelope3DParameters):
    """Hold the properties to compute transfer matrix of a :class:`.FieldMap`.

    Non-accelerating cavities will use :class:`.DriftEnvelope3DParameters`
    instead.

    """

    def __init__(self,
                 transf_mat_module: ModuleType,
                 elt: FieldMap,
                 n_steps: int,
                 method: str,
                 n_steps_per_cell: int,
                 solver_id: str,
                 phi_s_model: str = 'historical',
                 **kwargs: str,
                 ) -> None:
        """Create the specific parameters for a drift."""
        transf_mat_function = FIELD_MAP_INTEGRATION_METHOD_TO_FUNC[method](
            transf_mat_module)
        self.compute_cavity_parameters = \
            SYNCHRONOUS_PHASE_FUNCTIONS[phi_s_model]

        self.solver_id = solver_id
        self.n_cell = elt.get('n_cell')
        self.bunch_to_rf = elt.get('bunch_to_rf')
        n_steps = self.n_cell * n_steps_per_cell
        super().__init__(transf_mat_function,
                         elt.length_m,
                         n_steps,
                         )
        self._transf_mat_module = transf_mat_module
        elt.cavity_settings.set_beam_calculator(
            self.solver_id,
            self.transf_mat_function_wrapper
        )

    def transfer_matrix_arguments(self) -> tuple[float, int]:
        """Give the element parameters necessary to compute transfer matrix."""
        return self.d_z, self.n_steps

    def _transfer_matrix_results_to_dict(self,
                                         transfer_matrix: np.ndarray,
                                         gamma_phi: np.ndarray,
                                         itg_field: float | None,
                                         ) -> dict:
        """
        Convert the results given by the transf_mat function to dict.

        Overrides the default method defined in the ABC.

        """
        assert itg_field is not None
        w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
        gamma_phi[:, 1] /= self.bunch_to_rf
        cav_params = compute_param_cav(itg_field)
        results = {'transfer_matrix': transfer_matrix,
                   'r_zz': transfer_matrix[:, 4:, 4:],
                   'cav_params': cav_params,
                   'w_kin': w_kin,
                   'phi_rel': gamma_phi[:, 1]
                   }
        return results

    def re_set_for_broken_cavity(self) -> None:
        """Make beam calculator call Drift func instead of FieldMap."""
        self.transf_mat_function = self._transf_mat_module.drift

        def _new_transfer_matrix_results_to_dict(transfer_matrix: np.ndarray,
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
            results = {'transfer_matrix': transfer_matrix,
                       'r_zz': transfer_matrix[:, 4:, 4:],
                       'cav_params': cav_params,
                       'w_kin': w_kin,
                       'phi_rel': gamma_phi[:, 1]
                       }
            return results

        self._transfer_matrix_results_to_dict = \
            _new_transfer_matrix_results_to_dict


class BendEnvelope3DParameters(ElementEnvelope3DParameters):
    """Hold specific parameters to compute :class:`.Bend` transfer matrix."""

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
        raise NotImplementedError
