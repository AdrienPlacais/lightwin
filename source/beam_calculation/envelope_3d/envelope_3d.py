#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define :class:`Envelope3D`, an envelope solver."""
from dataclasses import dataclass

from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from core.accelerator.accelerator import Accelerator

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from beam_calculation.envelope_3d.single_element_envelope_3d_parameters import\
    SingleElementEnvelope3DParameters
from beam_calculation.envelope_3d.beam_parameters_factory import \
    BeamParametersFactoryEnvelope3D
from beam_calculation.envelope_3d.transfer_matrix_factory import \
    TransferMatrixFactoryEnvelope3D

from failures.set_of_cavity_settings import (SetOfCavitySettings,
                                             SingleCavitySettings)
from beam_calculation.envelope_3d.simulation_output_factory import \
    SimulationOutputFactoryEnvelope3D


@dataclass
class Envelope3D(BeamCalculator):
    """A 3D envelope solver."""

    flag_phi_abs: bool
    n_steps_per_cell: int

    def __post_init__(self):
        """Set the proper motion integration function, according to inputs."""
        self.out_folder += "_Envelope3D"
        super().__post_init__()

        self.beam_parameters_factory = BeamParametersFactoryEnvelope3D(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation
        )
        self.transfer_matrix_factory = TransferMatrixFactoryEnvelope3D(
            self.is_a_3d_simulation)

        import beam_calculation.envelope_3d.transfer_matrices_p as transf_mat
        self.transf_mat_module = transf_mat

    def _set_up_specific_factories(self) -> None:
        """Set up the factories specific to the :class:`.BeamCalculator`.

        This method is called in the :meth:`super().__post_init__`, hence it
        appears only in the base :class:`.BeamCalculator`.

        """
        self.simulation_output_factory = SimulationOutputFactoryEnvelope3D(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation,
            self.id,
            self.out_folder,
        )

    def run(self, elts: ListOfElements) -> SimulationOutput:
        """
        Compute beam propagation in 3D, envelope calculation.

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
        Envelope 3D calculation of beam in `elts`, with non-nominal settings.

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
            # FIXME
            gradient = None
            if 'grad' in elt.__dir__():
                gradient = elt.grad
            elt_results = \
                elt.beam_calc_param[self.id].transf_mat_function_wrapper(
                    w_kin,
                    elt.is_accelerating,
                    elt.get('status'),
                    gradient=gradient,
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
        Run Envelope3D with optimized cavity settings.

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
        kwargs = {
            'n_steps_per_cell': self.n_steps_per_cell,
            'transf_mat_module': self.transf_mat_module,
        }
        for elt in elts:
            elt.beam_calc_param[self.id] = SingleElementEnvelope3DParameters(
                elt,
                **kwargs)

        position = 0.
        index = 0
        for elt in elts:
            position, index = \
                elt.beam_calc_param[self.id].set_absolute_meshes(position,
                                                                 index)

    @property
    def is_a_multiparticle_simulation(self) -> bool:
        """Return False."""
        return False

    @property
    def is_a_3d_simulation(self) -> bool:
        """Return True."""
        return True

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


RF_FIELD_GETTERS = {
    'none': _no_rf_field,
    'failed': _no_rf_field,
    'nominal': _rf_field_kwargs_from_element,
    'rephased (ok)': _rf_field_kwargs_from_element,
    'compensate (ok)': _rf_field_kwargs_from_element,
    'compensate (not ok)': _rf_field_kwargs_from_element,
    'compensate (in progress)': _rf_field_from_single_cavity_settings,
}
