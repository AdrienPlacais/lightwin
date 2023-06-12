#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:24:37 2023.

@author: placais

TODO: access to listofelements._indiv_to_cumul_transf_mat
"""
import logging

from core.elements import _Element
from core.list_of_elements import ListOfElements
from core.emittance import beam_parameters_zdelta
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings


class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

    def __init__(self) -> None:
        """Create the object."""

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
        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        single_elts_results = []
        rf_fields = []
        for elt in elts:
            elt_results, rf_field = self._proper_transf_mat(elt, phi_abs,
                                                            w_kin)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    def run_with_this(self, set_of_cavity_settings: SetOfCavitySettings | dict,
                      elts: ListOfElements,
                      transfer_data: bool = True) -> SimulationOutput:
        """Perform a simulation with new cavity settings."""
        if isinstance(set_of_cavity_settings, SetOfCavitySettings):
            return self.new_run_with_this(set_of_cavity_settings, elts)
        logging.critical(f"old, {transfer_data = }")

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        single_elts_results = []
        rf_fields = []
        for elt in elts:
            # TODO transform the set_of_cavity_settings, will not work as is
            elt_results, rf_field = self._proper_transf_mat(
                elt, phi_abs, w_kin,
                set_of_cavity_settings=set_of_cavity_settings)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field)

            if (not transfer_data) and (elt.get('status') == 'nominal'):
                logging.critical('sa arrive sa des fois?')
                single_elts_results[-1]['phi_s'] = None
                rf_fields[-1] = {}

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    # FIXME I think it is possible to simplify all of this
    def _proper_transf_mat(self, elt: _Element, phi_abs: float, w_kin: float,
                           set_of_cavity_settings: dict | None = None
                           ) -> tuple[dict, dict]:
        """Get the proper arguments and call the elt.calc_transf_mat."""
        d_fits = set_of_cavity_settings
        d_fit_elt = None
        if elt.get('nature') == 'FIELD_MAP' and elt.get('status') != 'failed':
            # General case: take the nominal cavity parameters
            d_fit_elt = d_fits

            # Specific case of fit under process: extract new cavity parameters
            if d_fits is not None \
               and elt.get('status') == 'compensate (in progress)':
                d_fit_elt = {'flag': True,
                             'phi': d_fits['l_phi'].pop(0),
                             'k_e': d_fits['l_k_e'].pop(0),
                             'phi_s fit': d_fits['phi_s fit']}

        # Create an rf_field dict with data from d_fit_elt or element according
        # to the case
        rf_field_kwargs = elt.rf_param(phi_abs, w_kin, d_fit_elt)
        # Compute transf mat, acceleration, phase, etc
        elt_results = elt.calc_transf_mat(w_kin, **rf_field_kwargs)
        return elt_results, rf_field_kwargs

    def new_run_with_this(self,
                          cavities_settings: SetOfCavitySettings,
                          elts: ListOfElements,
                          ) -> SimulationOutput:
        """Compute the transfer matrices of Accelerator's elements."""
        single_elts_results = []
        rf_fields = []

        w_kin = elts.w_kin_in
        phi_abs = elts.phi_abs_in

        # Compute transfer matrix and acceleration in each element
        for elt in elts:
            cavity_settings = None
            if elt in cavities_settings:
                cavity_settings = cavities_settings[elt]

            rf_field_kwargs = elt.new_rf_param(phi_abs, w_kin, cavity_settings)
            elt_results = elt.calc_transf_mat(w_kin, **rf_field_kwargs)

            single_elts_results.append(elt_results)
            rf_fields.append(rf_field_kwargs)

            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        simulation_output = self._generate_simulation_output(
            elts, single_elts_results, rf_fields)
        return simulation_output

    # TODO only return what is needed for the fit?
    # maybe: a second specific _generate_simulation_output for fit only? This
    # is secondary for now...
    def _generate_simulation_output(self, elts: ListOfElements,
                                    single_elts_results: list[dict],
                                    rf_fields: list[dict | None]
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

        # FIXME
        mismatch_factor = None  # for results in single_elts_results]

        cav_params = [results['cav_params']
                      for results in single_elts_results]
        phi_s = [cav_param['phi_s']
                 for cav_param in cav_params if cav_param is not None]

        r_zz_elt = [
            results['r_zz'][i, :, :]
            for results in single_elts_results
            for i in range(results['r_zz'].shape[0])
        ]
        tm_cumul = elts._indiv_to_cumul_transf_mat(
            r_zz_elt, len(w_kin))

        beam_params = beam_parameters_zdelta(tm_cumul)

        simulation_output = SimulationOutput(
            w_kin=w_kin,
            phi_abs_array=phi_abs_array,
            mismatch_factor=mismatch_factor,
            cav_params=cav_params,
            phi_s=phi_s,
            r_zz_elt=r_zz_elt,
            tm_cumul=tm_cumul,
            rf_fields=rf_fields,
            eps_zdelta=beam_params[0],
            twiss_zdelta=beam_params[1],
            sigma_matrix=beam_params[2]
        )
        return simulation_output

    def _format_this(self, set_of_cavity_settings: SetOfCavitySettings
                     ) -> dict:
        """Transform `set_of_cavity_settings` for this BeamCalculator."""
        d_fit_elt = {}
        return d_fit_elt
