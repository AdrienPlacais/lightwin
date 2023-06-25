#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:24:37 2023.

@author: placais

"""
import logging
from typing import Callable
from types import ModuleType
from dataclasses import dataclass

import numpy as np

from core.particle import ParticleFullTrajectory
from core.elements import _Element
from core.list_of_elements import (ListOfElements, indiv_to_cumul_transf_mat)
from core.beam_parameters import BeamParameters
from core.electric_field import compute_param_cav
from beam_calculation.beam_calculator import (
    BeamCalculator, SingleElementCalculatorParameters)
from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings
import util.converters as convert


@dataclass
class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

    FLAG_PHI_ABS: bool
    FLAG_CYTHON: bool
    N_STEPS_PER_CELL: int
    METHOD: str

    def __post_init__(self):
        """Set the proper motion integration function, according to inputs."""
        if self.FLAG_CYTHON:
            try:
                import core.transfer_matrices_c as transf_mat
            except ModuleNotFoundError:
                logging.error("Cython version of transfer_matrices was not "
                              + "compiled. Check util/setup.py.")
                raise ModuleNotFoundError("Cython not compiled.")
        else:
            import core.transfer_matrices_p as transf_mat
        self.transf_mat_module = transf_mat

    def run(self, elts: ListOfElements) -> SimulationOutput:
        """
        Compute beam propagation in 1D, envelope calculation.

        Parameters
        ----------
        elts : ListOfElements
            List of elements in which the beam must be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        return self.run_with_this(set_of_cavity_settings=None, elts=elts)

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | None,
                      elts: ListOfElements) -> SimulationOutput:
        """
        Envelope 1D calculation of beam in `elts`, with non-nominal settings.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            The new cavity settings to try. If it is None, then the cavity
            settings are taken from the FieldMap objects.
        elts : ListOfElements
            List of elements in which the beam must be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        if True:
            return self.new_run_with_this(set_of_cavity_settings, elts)

        single_elts_results = []
        rf_fields = []

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in
        for elt in elts:
            cavity_settings = set_of_cavity_settings.get(elt) \
                if isinstance(set_of_cavity_settings, SetOfCavitySettings) \
                else None

            rf_field_kwargs = elt.rf_param(phi_abs, w_kin, cavity_settings)
            elt_results = elt.calc_transf_mat(w_kin, **rf_field_kwargs)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field_kwargs)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    def new_run_with_this(self,
                          set_of_cavity_settings: SetOfCavitySettings | None,
                          elts: ListOfElements) -> SimulationOutput:
        """
        Envelope 1D calculation of beam in `elts`, with non-nominal settings.

        Parameters
        ----------
        set_of_cavity_settings : SetOfCavitySettings | None
            The new cavity settings to try. If it is None, then the cavity
            settings are taken from the FieldMap objects.
        elts : ListOfElements
            List of elements in which the beam must be propagated.

        Returns
        -------
        simulation_output : SimulationOutput
            Holds energy, phase, transfer matrices (among others) packed into a
            single object.

        """
        single_elts_results = []
        rf_fields = []

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        for elt in elts:
            cavity_settings = set_of_cavity_settings.get(elt) \
                if isinstance(set_of_cavity_settings, SetOfCavitySettings) \
                else None

            rf_field_kwargs = elt.rf_param(phi_abs, w_kin, cavity_settings)
            elt_results = a_new_calc_transf_mat(w_kin, elt, **rf_field_kwargs)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field_kwargs)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    def _init_solver_parameters(self, elts: ListOfElements) -> None:
        """
        Create the number of steps, meshing, transfer functions for elts.

        The solver parameters are stored in self.parameters. As for now, for
        memory purposes, only one set of solver parameters is stored. In other
        words, if you compute the transfer matrices of several ListOfElements
        back and forth, the solver paramters will be re-initialized each time.

        Parameters
        ----------
        elts : ListOfElements
            The list of elements for which you want the parameters.

        """
        kwargs = {'n_steps_per_cell': self.N_STEPS_PER_CELL,
                  'method': self.METHOD,
                  'transf_mat_module': self.transf_mat_module,
                  }
        for elt in elts:
            elt.beam_calc_param = SingleElementEnvelope1DParameters(
                length_m=elt.get('length_m', to_numpy=False),
                is_accelerating=elt.is_accelerating(),
                n_cells=elt.get('n_cell', to_numpy=False),
                **kwargs)

        position = 0.
        index = 0
        for elt in elts:
            position, index = elt.beam_calc_param.set_absolute_meshes(position,
                                                                      index)

    def _generate_simulation_output(self, elts: ListOfElements,
                                    single_elts_results: list[dict],
                                    rf_fields: list[dict]
                                    ) -> SimulationOutput:
        """Transform the outputs of BeamCalculator to a SimulationOutput."""
        w_kin = [energy
                 for results in single_elts_results
                 for energy in results['w_kin']
                 ]
        w_kin.insert(0, elts.w_kin_in)

        phi_abs_array = [elts.phi_abs_in]
        for elt_results in single_elts_results:
            phi_abs = [phi_rel + phi_abs_array[-1]
                       for phi_rel in elt_results['phi_rel']]
            phi_abs_array.extend(phi_abs)
        synch_trajectory = ParticleFullTrajectory(w_kin=w_kin,
                                                  phi_abs=phi_abs_array,
                                                  synchronous=True)

        cav_params = [results['cav_params']
                      for results in single_elts_results]
        phi_s = [cav_param['phi_s']
                 for cav_param in cav_params if cav_param is not None]

        r_zz_elt = [
            results['r_zz'][i, :, :]
            for results in single_elts_results
            for i in range(results['r_zz'].shape[0])
        ]
        tm_cumul = indiv_to_cumul_transf_mat(elts.tm_cumul_in, r_zz_elt,
                                             len(w_kin))

        beam_params = BeamParameters(tm_cumul)

        element_to_index = self._generate_element_to_index_func(elts)

        simulation_output = SimulationOutput(
            synch_trajectory=synch_trajectory,
            cav_params=cav_params,
            phi_s=phi_s,
            r_zz_elt=r_zz_elt,
            rf_fields=rf_fields,
            beam_parameters=beam_params,
            element_to_index=element_to_index
        )
        return simulation_output

    # TODO update this once I remove s_in s_out from the Elements
    def _generate_element_to_index_func(self, elts: ListOfElements
                                        ) -> Callable[[_Element, str | None],
                                                      int | slice]:
        """Create the func to easily get data at proper mesh index."""
        shift = elts[0].get('s_in', to_numpy=False)

        def element_to_index(elt: _Element, pos: str | None = None
                             ) -> int | slice:
            """
            Convert element + pos into a mesh index.

            Parameters
            ----------
            elt : _Element
                Element of which you want the index.
            pos : 'in' | 'out' | None, optional
                Index of entry or exit of the _Element. If None, return full
                indexes array. The default is None.

            """
            if pos is None:
                return slice(elt.get('s_in', to_numpy=False) - shift,
                             elt.get('s_out', to_numpy=False) - shift + 1)
            elif pos == 'in':
                return elt.get('s_in', to_numpy=False) - shift
            elif pos == 'out':
                return elt.get('s_out', to_numpy=False) - shift
            else:
                logging.error(f"{pos = }, while it must be 'in', 'out' or "
                              + "None")
        return element_to_index

    def _format_this(self, set_of_cavity_settings: SetOfCavitySettings
                     ) -> dict:
        """Transform `set_of_cavity_settings` for this BeamCalculator."""
        d_fit_elt = {}
        return d_fit_elt

    def generate_set_of_cavity_settings(self, d_fit: dict
                                        ) -> SetOfCavitySettings:
        return None

    def init_all_meshes(self, elts: list[_Element]) -> None:
        """Init the mesh of every _Element (where quantities are evaluated)."""
        method = self.METHOD
        if '_' in method:
            method = method('_')[0]

        n_steps_setters = {
            'RK': lambda elt: self.N_STEPS_PER_CELL * elt.get('n_cell'),
            'leapfrog': lambda elt: self.N_STEPS_PER_CELL * elt.get('n_cell'),
            'drift': lambda elt: 1
        }

        for elt in elts:
            key_steps = method
            if elt.get('nature') != 'FIELD_MAP':
                key_steps = 'drift'
            n_steps = n_steps_setters[key_steps]

            elt.init_mesh(n_steps)

    def init_specific(self, elts: list[_Element]) -> None:
        """Set the proper transfer matrix function."""
        self._init_all_transfer_matrix_functions(elts)

    def _init_all_transfer_matrix_functions(self, elts: list[_Element]
                                            ) -> None:
        """Set the proper transfer matrix function."""
        method = self.METHOD
        if '_' in method:
            method = method('_')[0]

        transfer_matrix_function_setters = {
            'RK': self.transf_mat.z_field_map_rk4,
            'leapfrog': self.transf_mat.z_field_map_leapfrog,
            'drift': self.transf_mat.z_drift
        }

        for elt in elts:
            key_transf_mat = 'drift'
            is_accelerating = elt.get('nature') == 'FIELD_MAP' and \
                elt.get('status') != 'failed'
            if is_accelerating:
                key_transf_mat = method

            elt._tm_func = transfer_matrix_function_setters[key_transf_mat]


class SingleElementEnvelope1DParameters(SingleElementCalculatorParameters):
    """Holds the parameters to compute beam propagation in an _Element."""

    def __init__(self, length_m: float, is_accelerating: bool,
                 n_cells: int | None, n_steps_per_cell: int, method: str,
                 transf_mat_module: ModuleType) -> None:
        """Set the actually useful parameters."""
        self.n_steps = 1
        self.back_up_function = transf_mat_module.z_drift
        self.transf_mat_function = transf_mat_module.z_drift
        if is_accelerating:
            assert n_cells is not None
            self.n_steps = n_cells * n_steps_per_cell

            if 'RK' in method:
                self.transf_mat_function = transf_mat_module.z_field_map_rk4
            elif 'leapfrog' in method:
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


def a_new_calc_transf_mat(w_kin_in: float, elt: _Element, **rf_field_kwargs):
    """Let's try something."""
    gamma = convert.energy(w_kin_in, "kin to gamma")

    args = (elt.beam_calc_param.d_z, gamma, elt.beam_calc_param.n_steps)

    cav_params = None
    if elt.is_accelerating():
        r_zz, gamma_phi, itg_field = elt.beam_calc_param.transf_mat_function(
            *args, dict_rf_field=rf_field_kwargs)
        gamma_phi[:, 1] /= elt.get('n_cell', to_numpy=False)
        cav_params = compute_param_cav(itg_field, elt.get('status'))

    else:
        r_zz, gamma_phi, _ = elt.beam_calc_param.transf_mat_function(*args)

    w_kin = convert.energy(gamma_phi[:, 0], "gamma to kin")
    results = {'r_zz': r_zz, 'cav_params': cav_params,
               'w_kin': w_kin, 'phi_rel': gamma_phi[:, 1]}

    return results
