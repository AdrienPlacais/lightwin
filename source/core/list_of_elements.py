#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais
"""
import logging
from typing import Any
import numpy as np
from util.helper import recursive_items, recursive_getter
from core.emittance import beam_parameters_zdelta
from core.elements import _Element


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, l_elts: list[_Element], w_kin: float, phi_abs: float,
                 idx_in: int | None = None,
                 tm_cumul: np.ndarray | None = None) -> None:
        super().__init__(l_elts)
        logging.info(f"Init list from {l_elts[0].get('elt_name')} to "
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

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or \
            key in recursive_items(vars(self[0]))

    def get(self, *keys: tuple[str], to_numpy: bool = True,
            remove_first: bool = False, **kwargs: dict) -> Any:
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
    def compute_transfer_matrices(self, d_fits: dict | None = None,
                                  transfer_data: bool = True) -> dict:
        """
        Compute the transfer matrices of Accelerator's elements.

        Parameters
        ----------
        d_fits : dict, optional
            Dict to where norms and phases of compensating cavities are stored.
            If the dict is None, we take norms and phases from cavity objects.
            Default is None.
        transfer_data : boolean, optional
            If True, we save the energies, transfer matrices, etc that are
            calculated in the routine. Default is True.

        Returns
        -------
        results : dict
            Holds energy, phase, transfer matrices (among others) packed into a
            single dict.

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
    def _proper_transf_mat(self, elt: _Element, phi_abs: float, w_kin: float,
                           d_fits: dict) -> tuple[dict, dict]:
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
    def _pack_into_single_dict(self, l_elt_results: list[dict],
                               l_rf_fields: list[dict]) -> dict:
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
            "sigma matrix": None,
            # Mismatch has to be computed in another function, as it is
            # relative to another linac
            "mismatch factor": None,
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

        results["eps_zdelta"], results['twiss_zdelta'], \
            results["sigma matrix"] = beam_parameters_zdelta(
                results["tm_cumul"])
        return results

    def _indiv_to_cumul_transf_mat(self, l_r_zz_elt: list[np.ndarray],
                                   n_steps: int) -> np.ndarray:
        """Compute cumulated transfer matrix."""
        # Compute transfer matrix of l_elts
        arr_tm_cumul = np.full((n_steps, 2, 2), np.NaN)
        arr_tm_cumul[0] = self.tm_cumul_in
        for i in range(1, n_steps):
            arr_tm_cumul[i] = l_r_zz_elt[i - 1] @ arr_tm_cumul[i - 1]
        return arr_tm_cumul


def equiv_elt(elts: ListOfElements | list[_Element, ...],
              elt: _Element | str, to_index: bool = False
              ) -> _Element | int | None:
    """
    Return an element from elts that has the same name as elt.

    Important: this routine uses the name of the element and not its adress. So
    it will not complain if the _Element object that you asked for is not in
    this list of elements.
    In the contrary, it was meant to find equivalent cavities between different
    lists of elements.

    Parameters
    ----------
    elts : ListOfElements | list[_Element, ...]
        List of elements where you want the position.
    elt : _Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an _Element. If it is an _Element, we take its name in the
        routine.
    to_index : bool, optional
        If True, the function returns the index of the _Element instead of the
        _Element itself.

    Returns
    -------
    _Element | int | None
        Equivalent _Element, position in list of elements, or None if not
        found.

    """
    if not isinstance(elt, str):
        elt = elt.get("elt_name")

    names = [x.get("elt_name") for x in elts]
    if elt not in names:
        logging.error(f"Element {elt} not found in this list of elements.")
        logging.debug(f"List of elements is:\n{elts}")
        return None

    idx = names.index(elt)
    if to_index:
        return idx
    return elts[idx]


def elt_at_this_s_idx(elts: ListOfElements | list[_Element, ...],
                      s_idx: int, show_info: bool = False
                      ) -> _Element | None:
    """Give the element where the given index is."""
    for elt in elts:
        if s_idx in range(elt.idx['s_in'], elt.idx['s_out']):
            if show_info:
                logging.info(
                    f"Mesh index {s_idx} is in {elt.get('elt_info')}.\n"
                    + f"Indexes of this elt: {elt.get('idx')}.")
            return elt

    logging.warning(f"Mesh index {s_idx} not found.")
    return None


def get_elts(elts: ListOfElements | list[_Element, ...],
             key: str, val: Any) -> list[_Element, ...]:
    """
    Return a list of elements which key correspond to value.

    Example: sort_elts(list_of_elts, key='nature', val='FIELD_MAP')
    """
    return list(filter(lambda elt: elt.get(key) == val, elts))
