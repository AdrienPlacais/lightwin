#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module holds :class:`Envelope1D`, a longitudinal envelope solver.

It is fast, but should not be used at low energies.

"""
import logging
from typing import Any, Callable
from dataclasses import dataclass

import numpy as np

from core.particle import ParticleFullTrajectory
from core.elements.field_map import FieldMap
from core.elements.element import Element
from core.list_of_elements.list_of_elements import ListOfElements
from core.list_of_elements.helper import indiv_to_cumul_transf_mat
from core.accelerator import Accelerator
from core.beam_parameters.beam_parameters import BeamParameters
from core.transfer_matrix import TransferMatrix

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.output import SimulationOutput
from beam_calculation.envelope_1d.single_element_envelope_1d_parameters import\
    SingleElementEnvelope1DParameters

from failures.set_of_cavity_settings import (SetOfCavitySettings,
                                             SingleCavitySettings)
from util import converters


@dataclass
class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

    flag_phi_abs: bool
    flag_cython: bool
    n_steps_per_cell: int
    method: str

    def __post_init__(self):
        """Set the proper motion integration function, according to inputs."""
        self.id = self.__repr__()
        self.out_folder += "_Envelope1D"

        if self.flag_cython:
            try:
                import beam_calculation.envelope_1d.transfer_matrices_c as \
                    transf_mat
            except ModuleNotFoundError:
                logging.error("Cython version of transfer_matrices was not "
                              + "compiled. Check util/setup.py.")
                raise ModuleNotFoundError("Cython not compiled.")
        else:
            import beam_calculation.envelope_1d.transfer_matrices_p as \
                transf_mat
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
        single_elts_results = []
        rf_fields = []

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        for elt in elts:
            cavity_settings = set_of_cavity_settings.get(elt) \
                if isinstance(set_of_cavity_settings, SetOfCavitySettings) \
                else None

            rf_field_kwargs = elt.rf_param(self.id, phi_abs, w_kin,
                                           cavity_settings)
            elt_results = \
                elt.beam_calc_param[self.id].transf_mat_function_wrapper(
                    w_kin, elt.is_accelerating(), elt.get('status'),
                    **rf_field_kwargs)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field_kwargs)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    def post_optimisation_run_with_this(
        self,
        optimized_cavity_settings: SetOfCavitySettings,
        full_elts: ListOfElements,
        **specific_kwargs
    ) -> SimulationOutput:
        """
        Run Envelope1D with optimized cavity settings.

        With this solver, we have nothing to do, nothing to update. Just call
        the regular `run_with_this` method.

        """
        simulation_output = self.run_with_this(optimized_cavity_settings,
                                               full_elts,
                                               **specific_kwargs)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Create the number of steps, meshing, transfer functions for elts.

        The solver parameters are stored in self.parameters. As for now, for
        memory purposes, only one set of solver parameters is stored. In other
        words, if you compute the transfer matrices of several ListOfElements
        back and forth, the solver paramters will be re-initialized each time.

        Parameters
        ----------
        accelerator : Accelerator
            Accelerator object which ListOfElements must be initialized.

        """
        elts = accelerator.elts
        kwargs = {'n_steps_per_cell': self.n_steps_per_cell,
                  'method': self.method,
                  'transf_mat_module': self.transf_mat_module,
                  }
        for elt in elts:
            elt.beam_calc_param[self.id] = SingleElementEnvelope1DParameters(
                length_m=elt.get('length_m', to_numpy=False),
                is_accelerating=elt.is_accelerating(),
                n_cells=elt.get('n_cell', to_numpy=False),
                **kwargs)

        position = 0.
        index = 0
        for elt in elts:
            position, index = \
                elt.beam_calc_param[self.id].set_absolute_meshes(position,
                                                                 index)

    def _generate_simulation_output(self, elts: ListOfElements,
                                    single_elts_results: list[dict],
                                    rf_fields: list[dict]
                                    ) -> SimulationOutput:
        """
        Transform the outputs of BeamCalculator to a SimulationOutput.

        .. todo::
            Patch in transfer matrix to get proper input transfer matrix. In
            future, input beam will not hold transf mat in anymore.

        """
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
        gamma_kin = synch_trajectory.gamma
        assert isinstance(gamma_kin, np.ndarray)

        cav_params = [results['cav_params'] for results in single_elts_results]
        cav_params = {'v_cav_mv': [cav_param['v_cav_mv']
                                   if cav_param is not None else None
                                   for cav_param in cav_params],
                      'phi_s': [cav_param['phi_s']
                                if cav_param is not None else None
                                for cav_param in cav_params],
                      }

        element_to_index = self._generate_element_to_index_func(elts)
        transfer_matrix = _transfer_matrix_factory(
            elts.input_beam.zdelta.tm_cumul_in,
            single_elts_results
        )
        beam_parameters = _beam_parameters_factory(
            z_abs=elts.get('abs_mesh', remove_first=True),
            gamma_kin=gamma_kin,
            element_to_index=element_to_index,
            transfer_matrix=transfer_matrix,
            sigma_in=elts.input_beam.sigma_in
        )

        simulation_output = SimulationOutput(
            out_folder=self.out_folder,
            is_multiparticle=self.is_a_multiparticle_simulation,
            is_3d=self.is_a_3d_simulation,
            synch_trajectory=synch_trajectory,
            cav_params=cav_params,
            rf_fields=rf_fields,
            beam_parameters=beam_parameters,
            element_to_index=element_to_index,
            transfer_matrix=transfer_matrix
        )
        return simulation_output

    @property
    def is_a_multiparticle_simulation(self) -> bool:
        """Return False."""
        return False

    @property
    def is_a_3d_simulation(self) -> bool:
        """Return False."""
        return False

    def _proper_rf_field_kwards(
        self,
        cavity: FieldMap,
        new_cavity_settings: SingleCavitySettings | None = None,
    ) -> dict:
        """Return the proper rf field according to the cavity status."""
        rf_param = {
            'omega0_rf': cavity.get('omega0_rf'),
            'e_spat': cavity.acc_field.e_spat,
            'section_idx': cavity.idx['section'],
            'n_cell': cavity.get('n_cell')
        }

        getter = RF_FIELD_GETTERS[cavity.status]
        rf_param = rf_param | getter(cavity=cavity,
                                     new_cavity_settings=new_cavity_settings)
        return rf_param


def _no_rf_field(*args, **kwargs) -> dict:
    """Return an empty dict."""
    return {}


def _rf_field_kwargs_from_element(cavity: FieldMap) -> dict:
    """Get the data from the `FieldMap` object."""
    rf_param = {
        'k_e': cavity.acc_field.k_e,
        'phi_0_rel': None,
        'phi_0_abs': cavity.acc_field.phi_0_abs,
    }
    return rf_param


def _rf_field_from_single_cavity_settings(
    new_cavity_settings: SingleCavitySettings
) -> dict:
    """Get the data from the `SingleCavitySettings` object."""


def _transfer_matrix_factory(
        first_cumulated_transfer_matrix: np.ndarray,
        single_elts_results: list[dict[str, Any]]
        ) -> TransferMatrix:
    """Create the :class:`.TransferMatrix` from current solver results.

    Parameters
    ----------
    first_cumulated_transfer_matrix : np.ndarray
        Cumulated transfer matrix at beginning of :class:`.ListOfElements`
        under study.
    single_elts_results : list[dict[str, Any]]
        Results of the solver.

    Returns
    -------
    transfer_matrix : TransferMatrix
        Object holding transfer matrices, individual and cumulated, in the
        relatable phase-spaces.

    """
        # individual_1d = [results['r_zz'][i, :, :]
        #                  for results in single_elts_results
        #                  for i in range(results['r_zz'].shape[0])]
        # transfer_matrix = TransferMatrix(individual=individual_1d)
        # r_zz_elt = [results['r_zz'][i, :, :]
        #             for results in single_elts_results
        #             for i in range(results['r_zz'].shape[0])]
        # tm_cumul = indiv_to_cumul_transf_mat(elts.tm_cumul_in,
        #                                      r_zz_elt,
        #                                      len(w_kin))

    individual = [results['r_zz'][i]
                  for results in single_elts_results
                  for i in range(results['r_zz'].shape[0])]
    # individual = [results['transfer_matrix'][i]
    #               for results in single_elts_results
    #               for i in range(results['transfer_matrix'].shape[0])]

    if first_cumulated_transfer_matrix.shape == (2, 2):
        logging.warning("Here I should initialize TransferMatrix with "
                        "an initial transfer matrix, but I have a shape "
                        "mismatch. It is ok for now.")

    transfer_matrix = TransferMatrix(
        individual=individual,
        first_cumulated_transfer_matrix=first_cumulated_transfer_matrix
        )
    return transfer_matrix


def _beam_parameters_factory(
        z_abs: np.ndarray,
        gamma_kin: np.ndarray,
        element_to_index: Callable[[str | Element, str | None], int | slice],
        transfer_matrix: TransferMatrix,
        sigma_in: np.ndarray,
        ) -> BeamParameters:
    r"""Create the :class:`.BeamParameters` object from simulation results.

    Parameters
    ----------
    z_abs : np.ndarray
        Absolute position in the linac or the linac portion.
    gamma_kin : np.ndarray
        Lorentz factor.
    element_to_index : Callable[[str | Element, str | None], int | slice]
        Takes an :class:`.Element`, its name, 'first' or 'last' as argument,
        and returns corresponding index. Index should be the same in all the
        arrays attributes of this class: ``z_abs``, ``beam_parameters``
        attributes, etc.  Used to easily `get` the desired properties at the
        proper position.
    transfer_matrix : TransferMatrix
        Object holding transfer matrices.

    Returns
    -------
    beam_parameters : BeamParameters
        Object holding emittances, :math:`\sigma` beam matrices, Courant-Snyder
        parameters in the different phase-spaces.

    """
        # beam_params = BeamParameters(
        #     z_abs=elts.get('abs_mesh', remove_first=True),
        #     gamma_kin=synch_trajectory.gamma,
        #     element_to_index=element_to_index)
        # beam_params.create_phase_spaces('zdelta', 'z', 'phiw')
        # beam_params.zdelta.init_from_cumulated_transfer_matrices(
        #     gamma_kin=beam_params.gamma_kin,
        #     tm_cumul=tm_cumul,
        #     beta_kin=beam_params.beta_kin
        # )
        # beam_params.init_other_phase_spaces_from_zdelta(*('phiw', 'z'))

    beta_kin = converters.energy(gamma_kin, 'gamma to beta')
    assert isinstance(beta_kin, np.ndarray)

    beam_parameters = BeamParameters(z_abs=z_abs,
                                     gamma_kin=gamma_kin,
                                     beta_kin=beta_kin,
                                     element_to_index=element_to_index,
                                     sigma_in=sigma_in)

    beam_parameters.create_phase_spaces('z', 'zdelta', 'phiw')

    for phase_space_name, sub_transfer_matrix_name in zip(
            ('zdelta',),
            ('r_zdelta',)):
        phase_space = beam_parameters.get(phase_space_name)
        sub_transfer_matrix = transfer_matrix.get(sub_transfer_matrix_name)
        phase_space.init_from_cumulated_transfer_matrices(
            tm_cumul=sub_transfer_matrix,
            gamma_kin=gamma_kin,
            beta_kin=beta_kin
            )

    beam_parameters.init_other_phase_spaces_from_zdelta(*('phiw', 'z'))
    return beam_parameters


RF_FIELD_GETTERS = {
    'none': _no_rf_field,
    'failed': _no_rf_field,
    'nominal': _rf_field_kwargs_from_element,
    'rephased (ok)': _rf_field_kwargs_from_element,
    'compensate (ok)': _rf_field_kwargs_from_element,
    'compensate (not ok)': _rf_field_kwargs_from_element,
    'compensate (in progress)': _rf_field_from_single_cavity_settings,
}
