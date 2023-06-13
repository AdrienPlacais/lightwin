#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 08:24:37 2023.

@author: placais

TODO: access to listofelements._indiv_to_cumul_transf_mat
TODO: run and run_with_this have too much in common, should mutualize
"""
from core.list_of_elements import ListOfElements
from core.emittance import beam_parameters_zdelta
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.output import SimulationOutput
from optimisation.set_of_cavity_settings import SetOfCavitySettings


class Envelope1D(BeamCalculator):
    """The fastest beam calculator, adapted to high energies."""

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
            cavity_settings = None
            if (set_of_cavity_settings is not None
                    and elt in set_of_cavity_settings):
                cavity_settings = set_of_cavity_settings[elt]

            rf_field_kwargs = elt.rf_param(phi_abs, w_kin, cavity_settings)
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

    def generate_set_of_cavity_settings(self, d_fit: dict) -> SetOfCavitySettings:
        return None
