"""Define a class to easily generate the :class:`.SimulationOutput`."""

from abc import ABCMeta
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from beam_calculation.envelope_3d.beam_parameters_factory import (
    BeamParametersFactoryEnvelope3D,
)
from beam_calculation.envelope_3d.transfer_matrix_factory import (
    TransferMatrixFactoryEnvelope3D,
)
from beam_calculation.simulation_output.factory import SimulationOutputFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.list_of_elements.list_of_elements import ListOfElements
from core.particle import ParticleFullTrajectory
from failures.set_of_cavity_settings import SetOfCavitySettings


@dataclass
class SimulationOutputFactoryEnvelope3D(SimulationOutputFactory):
    """A class for creating simulation outputs for :class:`.Envelope3D`."""

    out_folder: Path

    def __post_init__(self) -> None:
        """Create the factories.

        The created factories are :class:`.TransferMatrixFactory` and
        :class:`.BeamParametersFactory`. The sub-class that is used is declared
        in :meth:`._transfer_matrix_factory_class` and
        :meth:`._beam_parameters_factory_class`.

        """
        # Factories created in ABC's __post_init__
        return super().__post_init__()

    @property
    def _transfer_matrix_factory_class(self) -> ABCMeta:
        """Give the **class** of the transfer matrix factory."""
        return TransferMatrixFactoryEnvelope3D

    @property
    def _beam_parameters_factory_class(self) -> ABCMeta:
        """Give the **class** of the beam parameters factory."""
        return BeamParametersFactoryEnvelope3D

    def run(
        self,
        elts: ListOfElements,
        single_elts_results: list[dict],
        set_of_cavity_settings: SetOfCavitySettings,
    ) -> SimulationOutput:
        """
        Transform the outputs of BeamCalculator to a SimulationOutput.

        .. todo::
            Patch in transfer matrix to get proper input transfer matrix. In
            future, input beam will not hold transf mat in anymore.

        """
        w_kin = [
            energy
            for results in single_elts_results
            for energy in results["w_kin"]
        ]
        w_kin.insert(0, elts.w_kin_in)

        phi_abs_array = [elts.phi_abs_in]
        for elt_results in single_elts_results:
            phi_abs = [
                phi_rel + phi_abs_array[-1]
                for phi_rel in elt_results["phi_rel"]
            ]
            phi_abs_array.extend(phi_abs)
        synch_trajectory = ParticleFullTrajectory(
            w_kin=w_kin, phi_abs=phi_abs_array, synchronous=True
        )
        gamma_kin = synch_trajectory.gamma
        assert isinstance(gamma_kin, np.ndarray)

        cav_params = {
            "v_cav_mv": [
                (
                    set_of_cavity_settings[elt].v_cav_mv
                    if elt in set_of_cavity_settings
                    else None
                )
                for elt in elts
            ],
            "phi_s": [
                (
                    set_of_cavity_settings[elt].phi_s
                    if elt in set_of_cavity_settings
                    else None
                )
                for elt in elts
            ],
        }

        element_to_index = self._generate_element_to_index_func(elts)
        transfer_matrix = self.transfer_matrix_factory.run(
            elts.tm_cumul_in, single_elts_results, element_to_index
        )

        z_abs = elts.get("abs_mesh", remove_first=True)
        beam_parameters = self.beam_parameters_factory.factory_method(
            elts.input_beam.sigma,
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
            beam_parameters=beam_parameters,
            element_to_index=element_to_index,
            transfer_matrix=transfer_matrix,
            set_of_cavity_settings=set_of_cavity_settings,
        )

        return simulation_output
