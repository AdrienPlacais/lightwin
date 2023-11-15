#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define a class to easily generate the :class:`.SimulationOutput."""
from dataclasses import dataclass
import numpy as np

from beam_calculation.simulation_output.factory import SimulationOutputFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
    TransferMatrix
)

from core.list_of_elements.list_of_elements import ListOfElements
from core.particle import ParticleFullTrajectory


@dataclass
class SimulationOutputFactoryEnvelope3D(SimulationOutputFactory):
    """A class for creating simulation outputs for :class:`.Envelope3D`."""

    out_folder: str

    def run(self,
            elts: ListOfElements,
            single_elts_results: list[dict],
            rf_fields: list[dict],
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
        transfer_matrix = self.transfer_matrix_factory.run(
            elts.tm_cumul_in,
            single_elts_results,
            element_to_index)

        z_abs = elts.get('abs_mesh', remove_first=True)
        beam_parameters = self.beam_parameters_factory.factory_method(
                elts.input_beam.sigma_in,
                z_abs,
                gamma_kin,
                transfer_matrix,
                element_to_index,
            )

        simulation_output = SimulationOutput(
            out_folder=self.out_folder,
            is_multiparticle=False,  # FIXME
            is_3d=True,
            synch_trajectory=synch_trajectory,
            cav_params=cav_params,
            rf_fields=rf_fields,
            beam_parameters=beam_parameters,
            element_to_index=element_to_index,
            transfer_matrix=transfer_matrix
        )

        return simulation_output
