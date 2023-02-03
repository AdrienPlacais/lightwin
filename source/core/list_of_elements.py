#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais
"""
import numpy as np
from util.helper import recursive_items, recursive_getter
from core.emittance import beam_parameters_zdelta


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, l_elts, w_kin, phi_abs, idx_in=None, tm_cumul=None):
        super().__init__(l_elts)
        print(f"Init list from {l_elts[0].get('elt_name')} to "
              f"{l_elts[-1].get('elt_name')}.")

        if idx_in is None:
            idx_in = l_elts[0].idx['s_in']
        self._idx_in = idx_in
        self.w_kin_in = w_kin
        self.phi_abs_in = phi_abs

        if self._idx_in == 0:
            tm_cumul = np.eye(2)
        else:
            assert ~np.isnan(tm_cumul).any(), \
                "Previous transfer matrix was not calculated."
        self.tm_cumul_in = tm_cumul

    def has(self, key):
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or \
            key in recursive_items(vars(self[0]))

    def get(self, *keys, to_numpy=True, remove_first=False, **kwargs):
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            # Specific case: key is in Element so we concatenate all
            # corresponding values in a single list
            if self[0].has(key):
                for elt in self:
                    data = elt.get(key, to_numpy=False, **kwargs)
                    # In some arrays such as z position arrays, the last pos of
                    # an element is the first of the next
                    if remove_first and elt is not self[0]:
                        data = data[1:]
                    if isinstance(data, list):
                        val[key] += data
                    else:
                        val[key].append(data)
            else:
                val[key] = recursive_getter(key, vars(self), **kwargs)

        # Convert to list, and to numpy array if necessary
        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) for val in out]

        # Return as tuple or single value
        if len(keys) == 1:
            return out[0]
        # implicit else
        return tuple(out)

    # FIXME transfer_data does not have the same meaning anymore
    def compute_transfer_matrices(self, d_fits=None, transfer_data=True):
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        d_fits: dict, optional
            Dict to where norms and phases of compensating cavities are stored.
            If the dict is None, we take norms and phases from cavity objects.
        transfer_data : boolean, optional
            If True, we save the energies, transfer matrices, etc that are
            calculated in the routine.
        """
        # Prepare lists to store each element's results
        l_elt_results = []
        l_rf_fields = []

        # Initial phase and energy values:
        w_kin = self.w_kin_in
        phi_abs = self.phi_abs_in

        # Compute transfer matrix and acceleration in each element
        for elt in self:
            elt_results, rf_field = \
                self._proper_transf_mat(elt, phi_abs, w_kin, d_fits)

            # Store this element's results
            l_elt_results.append(elt_results)
            l_rf_fields.append(rf_field)

            # If there is nominal cavities in the recalculated zone during a
            # fit, we remove the associated rf fields and phi_s
            # (not used by the optimisation algorithms)
            # FIXME simpler equivalent?
            if (not transfer_data) and (d_fits is not None) \
               and (elt.get('status') == 'nominal'):
                l_rf_fields[-1] = {}
                l_elt_results[-1]['phi_s'] = None

            # Update energy and phase
            phi_abs += elt_results["phi_rel"][-1]
            w_kin = elt_results["w_kin"][-1]

        # We store all relevant data in results: evolution of energy, phase,
        # transfer matrices, emittances, etc
        results = self._pack_into_single_dict(l_elt_results, l_rf_fields)
        return results

    # FIXME I think it is possible to simplify all of this
    def _proper_transf_mat(self, elt, phi_abs, w_kin, d_fits):
        """Get the proper arguments and call the elt.calc_transf_mat."""
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

    # FIXME could be simpler
    def _pack_into_single_dict(self, l_elt_results, l_rf_fields):
        """
        We store energy, transfer matrices, phase, etc into the results dict.

        This dict is used in the fitting process.
        """
        # To store results
        results = {
            "phi_s": [],
            "cav_params": [],
            "w_kin": [self.w_kin_in],
            "phi_abs_array": [self.phi_abs_in],
            "r_zz_elt": [],         # List of numpy arrays
            "tm_cumul": None,     # (n, 2, 2) numpy array
            "rf_fields": [],        # List of dicts
            "eps_zdelta": None,
            "twiss_zdelta": None,
        }

        for elt_results, rf_field in zip(l_elt_results, l_rf_fields):
            results["rf_fields"].append(rf_field)
            results["cav_params"].append(elt_results["cav_params"])
            if rf_field != {}:
                results["phi_s"].append(elt_results['cav_params']['phi_s'])

            r_zz_elt = [elt_results['r_zz'][i, :, :]
                        for i in range(elt_results['r_zz'].shape[0])]
            results["r_zz_elt"].extend(r_zz_elt)

            l_phi_abs = [phi_rel + results["phi_abs_array"][-1]
                         for phi_rel in elt_results['phi_rel']]
            results["phi_abs_array"].extend(l_phi_abs)

            results["w_kin"].extend(elt_results['w_kin'].tolist())

        results["tm_cumul"] = self._indiv_to_cumul_transf_mat(
            results["r_zz_elt"], len(results["w_kin"]))

        results["eps_zdelta"], results['twiss_zdelta'] = \
            beam_parameters_zdelta(results["tm_cumul"])
        return results

    def _indiv_to_cumul_transf_mat(self, l_r_zz_elt, n_steps):
        """Compute cumulated transfer matrix."""
        # Compute transfer matrix of l_elts
        arr_tm_cumul = np.full((n_steps, 2, 2), np.NaN)
        arr_tm_cumul[0] = self.tm_cumul_in
        for i in range(1, n_steps):
            arr_tm_cumul[i] = l_r_zz_elt[i - 1] @ arr_tm_cumul[i - 1]
        return arr_tm_cumul
