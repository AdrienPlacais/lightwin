#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calculate beam in 1D envelope.

It is fast, but should not be used at low energies.

"""
import logging
from pathlib import Path

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.envelope_1d.element_envelope1d_parameters_factory import\
    ElementEnvelope1DParametersFactory
from beam_calculation.envelope_1d.simulation_output_factory import \
    SimulationOutputFactoryEnvelope1D
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.elements.element import Element
from core.elements.field_maps.cavity_settings import CavitySettings
from core.elements.field_maps.field_map import FieldMap
from core.list_of_elements.list_of_elements import ListOfElements
from failures.set_of_cavity_settings import SetOfCavitySettings
from util.synchronous_phases import SYNCHRONOUS_PHASE_FUNCTIONS


class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

    def __init__(self,
                 flag_phi_abs: bool,
                 flag_cython: bool,
                 n_steps_per_cell: int,
                 method: str,
                 out_folder: Path | str,
                 default_field_map_folder: Path | str,
                 phi_s_definition: str = 'historical',
                 ) -> None:
        """Set the proper motion integration function, according to inputs."""
        self.flag_phi_abs = flag_phi_abs
        self.flag_cython = flag_cython
        self.n_steps_per_cell = n_steps_per_cell
        self.method = method
        super().__init__(out_folder=out_folder,
                         default_field_map_folder=default_field_map_folder,
                         )
        self._phi_s_definition = phi_s_definition
        self._phi_s_func = SYNCHRONOUS_PHASE_FUNCTIONS[self._phi_s_definition]

    def _set_up_specific_factories(self) -> None:
        """Set up the factories specific to the :class:`.BeamCalculator`.

        This method is called in the :meth:`super().__post_init__`, hence it
        appears only in the base :class:`.BeamCalculator`.

        """
        self.simulation_output_factory = SimulationOutputFactoryEnvelope1D(
            self.is_a_3d_simulation,
            self.is_a_multiparticle_simulation,
            self.id,
            self.out_folder,
        )
        self.beam_calc_parameters_factory = \
            ElementEnvelope1DParametersFactory(
                self.method,
                self.n_steps_per_cell,
                self.id,
                self.flag_cython,
            )

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
        Envelope 1D calculation of beam in ``elts``, with non-nominal settings.

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
            rf_field_kwargs = self._proper_cavity_settings(
                elt, set_of_cavity_settings, phi_abs, w_kin)

            elt_results = \
                elt.beam_calc_param[self.id].transf_mat_function_wrapper(
                    w_kin,
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
        """Run :class:`Envelope1D. with optimized cavity settings.

        With this solver, we have nothing to do, nothing to update. Just call
        the regular :meth:`run_with_this` method.

        """
        simulation_output = self.run_with_this(optimized_cavity_settings,
                                               full_elts,
                                               **specific_kwargs)
        return simulation_output

    def init_solver_parameters(self, accelerator: Accelerator) -> None:
        """
        Create the number of steps, meshing, transfer functions for elts.

        The solver parameters are stored in ``self.parameters``. As for now,
        for memory purposes, only one set of solver parameters is stored.
        In other words, if you compute the transfer matrices of several
        :class:`.ListOfElements` back and forth, the solver paramters will be
        re-initialized each time.

        Parameters
        ----------
        accelerator : Accelerator
            Object which :class:`.ListOfElements` must be initialized.

        """
        elts = accelerator.elts
        for elt in elts:
            solver_param = self.beam_calc_parameters_factory.run(elt)
            elt.beam_calc_param[self.id] = solver_param

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
        """Return False."""
        return False

    def _proper_cavity_settings(
            self,
            element: Element,
            set_of_cavity_settings: SetOfCavitySettings | None,
            *args,
            **kwargs) -> dict:
        """Take proper :class:`.CavitySettings`, format it for solver."""
        if not isinstance(element, FieldMap):
            return {}
        if element.elt_info['status'] == 'failed':
            return {}

        cavity_settings = element.cavity_settings
        if (set_of_cavity_settings is not None
                and element in set_of_cavity_settings):
            cavity_settings = set_of_cavity_settings.get(element)
        assert isinstance(cavity_settings, CavitySettings), (
            f"{type(cavity_settings) = }")

        return self._adapt_cavity_settings(element,
                                           cavity_settings,
                                           *args,
                                           **kwargs)

    def _adapt_cavity_settings(self,
                               field_map: FieldMap,
                               cavity_settings: CavitySettings,
                               phi_bunch_abs: float,
                               w_kin_in: float,
                               *args,
                               **kwargs) -> dict:
        """Format the given :class:`.CavitySettings` for current solver.

        For the transfer matrix function of :class:`Envelope1D`, we need a
        dictionary.

        """
        if cavity_settings.status == 'none':
            logging.critical("Does 'none' status exists?")
            return {}
        if cavity_settings.status == 'failed':
            return {}

        cavity_settings.phi_bunch = phi_bunch_abs

        rf_parameters_as_dict = {
            'omega0_rf': field_map.get('omega0_rf'),
            'e_spat': field_map.rf_field.e_spat,
            'section_idx': field_map.idx['section'],
            'n_cell': field_map.get('n_cell'),
            # old implementation
            'bunch_to_rf': field_map.get('bunch_to_rf'),
            # future implementation
            # 'bunch_to_rf_func': cavity_settings._bunch_phase_to_rf_phase,
            'phi_0_rel': cavity_settings.phi_0_rel,
            'phi_0_abs': cavity_settings.phi_0_abs,
            'k_e': cavity_settings.k_e,
        }
        cavity_settings.set_phi_s_calculators(self.id, w_kin_in,
                                              **rf_parameters_as_dict)
        return rf_parameters_as_dict
