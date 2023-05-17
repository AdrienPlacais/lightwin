#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022.

@author: placais

Module holding the FaultScenario, which holds the Faults. Each Fault object
fixes himself (Fault.fix).

brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.

TODO in neighboring_cavities, allow for preference of cavities according to
their section

TODO allow for different strategies according to the section
TODO raise explicit error when the format of error (list vs idx) is not
appropriate, especially when manual mode.

TODO tune the PSO
TODO method to avoid big changes of acceptance
TODO option to minimize the power of compensating cavities

TODO remake a small fit after the first one?
"""
import logging
import itertools
import math
import numpy as np
import pandas as pd

import config_manager as con
import optimisation.fault as mod_f
from core.accelerator import Accelerator
from core.list_of_elements import ListOfElements
from core.elements import FieldMap
from core.emittance import mismatch_factor
from util import debug


# Clarify this shit
class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac: Accelerator, broken_linac: Accelerator,
                 wtf: dict,
                 l_fault_idx: list[int, ...] | list[list[int, ...], ...],
                 l_comp_idx: list[list[int, ...], ...] | None = None,
                 l_info_other_sol: list[dict, ...] = None) -> None:
        """
        Create the FaultScenario and the Faults.

        Parameters
        ----------
        ref_linac : Accelerator
            Reference linac.
        broken_linac : Accelerator
            Linac to fix.
        wtf : dict
            Holds what to fit.
        l_fault_idx : list
            List containing the position of the errors. If strategy is manual,
            it is a list of lists (faults already gathered).
        l_comp_idx : list, optional
            List containing the position of the compensating cavities. If
            strategy is manual, it must be provided. The default is None.
        l_info_other_sol : list, optional
            Contains information on another fit, for comparison purposes. The
            default is None.

        """
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac
        self.wtf = wtf
        self.l_info_other_sol = l_info_other_sol

        assert ref_linac.get('reference') and not broken_linac.get('reference')
        self.brok_lin.name += f" {str(l_fault_idx)}"

        ll_comp_cav, ll_fault_idx = self._sort_faults(l_fault_idx, l_comp_idx)
        l_obj = self._create_fault_objects(ll_fault_idx, ll_comp_cav)

        self.faults = {
            # List of Fault objects
            'l_obj': l_obj,
            # List of list of index of failed cavities, grouped by Fault
            'l_idx': ll_fault_idx,
            # List of list of compensating + failed cavities, grouped by Fault
            'l_comp': ll_comp_cav,
        }

        # Ensure that the cavities are sorted from linac entrance to linac exit
        flattened_l_fault_idx = [idx
                                 for l_idx in ll_fault_idx
                                 for idx in l_idx]
        assert flattened_l_fault_idx == sorted(flattened_l_fault_idx)

        self.info = {'fit': None}

        # Change status of cavities after the first failed cavity to tell LW
        # that they must keep their relative entry phases, not their absolute
        if not con.FLAG_PHI_ABS:
            self._update_status_of_cavities_to_rephase()
        self._transfer_phi0_from_ref_to_broken()

        results = self.brok_lin.elts.compute_transfer_matrices()
        results["mismatch factor"] = self._compute_mismatch()
        self.brok_lin.store_results(results, self.brok_lin.elts)

    # TODO clarify this shit also
    def _sort_faults(self,
                     l_fault_idx: list[int, ...] | list[list[int, ...], ...],
                     ll_comp_idx: list[list[int, ...], ...] | None,
                     ) -> tuple[list[list[FieldMap, ...], ...],
                                list[list[int, ...], ...]]:
        """
        Gather faults that are close to each other.

        Parameters
        ----------
        l_fault_idx : list[int, ...] | list[list[int, ...], ...]
            List of the indexes of the faulty cavities. If strategy is manual,
            they are already grouped.
        ll_comp_idx : list[list[int, ...], ...] | None
            If strategy is not manual, this is None. If strategy is manual,
            this is the list of compensating cavities, grouped for l_fault_idx.

        Returns
        -------
        ll_comp : list[list[FieldMap, ...], ...]
            List of list of compensating cavities. The length of the outer list
            is the same as l_fault_idx.
        l_fault_idx : list[list[int, ...], ...]
            List of list of faulty cavities. They are gathered by Faults to be
            fixed by common compensating cavities.

        """
        lin = self.brok_lin

        # Initialize list of list of faults indexes (provided by user if in
        # manual strategy)
        ll_fault_idx = l_fault_idx
        if self.wtf['strategy'] != 'manual':
            ll_fault_idx = [[idx] for idx in sorted(l_fault_idx)]

        # Get the absolute position of the FIELD_MAPS
        if self.wtf['idx'] == 'cavity':
            # l_cav = lin.get_elts('nature', 'FIELD_MAP')
            ll_fault_idx = [
                [lin.l_cav[cav].get('elt_idx', to_numpy=False) for cav in i]
                for i in ll_fault_idx]

            if ll_comp_idx is not None:
                ll_comp_idx = [
                    [lin.l_cav[cav].get('elt_idx', to_numpy=False)
                     for cav in i]
                    for i in ll_comp_idx]

        # If in manual mode, faults should be already gathered
        if self.wtf['strategy'] == 'manual':
            ll_comp = manually_set_cavities(lin, ll_fault_idx, ll_comp_idx)

        else:
            # Initialize list of list of corresp. faulty cavities
            ll_faults = [[lin.elts[idx]
                          for idx in l_idx]
                         for l_idx in ll_fault_idx]

            # We go across all faults and determine the compensating cavities
            # they need. If two failed cavities need at least one compensating
            # cavity in common, we group them together.
            # In particular, it is the case when a full cryomodule fails.
            ll_comp, ll_fault_idx = self._gather(ll_faults, ll_fault_idx)

        return ll_comp, ll_fault_idx

    def _gather(self, ll_faults, ll_idx_faults):
        """Proper method that gathers faults requiring the same compens cav."""
        d_comp_cav = {
            'k out of n': lambda l_cav:
                neighboring_cavities(self.brok_lin, l_cav, self.wtf['k']),
            'l neighboring lattices': lambda l_cav:
                neighboring_lattices(self.brok_lin, l_cav, self.wtf['l']),
            'global': lambda l_cav:
                all_cavities_after_first_fault(self.brok_lin, l_cav)}
        flag_gathered = False
        r_comb = 2

        while not flag_gathered:
            # List of list of corresp. compensating cavities
            ll_comp = [d_comp_cav[self.wtf['strategy']](l_cav)
                       for l_cav in ll_faults]

            # Set a counter to exit the 'for' loop when all faults are gathered
            i = 0
            n_comb = len(ll_comp)
            if n_comb <= 1:
                flag_gathered = True
                break

            i_max = int(math.factorial(n_comb) / (
                math.factorial(r_comb) * math.factorial(n_comb - r_comb)))

            # Now we look every list of required compensating cavities, and
            # look for faults that require the same compensating cavities
            for ((idx1, l_comp1), (idx2, l_comp2)) \
                in itertools.combinations(enumerate(ll_comp),
                                          r_comb):
                i += 1
                common_cav = list(set(l_comp1) & set(l_comp2))
                # If at least one cavity on common, gather the two
                # corresponding fault and restart the whole process
                if len(common_cav) > 0:
                    ll_faults[idx1].extend(ll_faults.pop(idx2))
                    ll_idx_faults[idx1].extend(ll_idx_faults.pop(idx2))
                    break

                # If we reached this point, it means that there is no list of
                # faults that share compensating cavities.
                if i == i_max:
                    flag_gathered = True

        return ll_comp, ll_idx_faults

    def _create_fault_objects(self, ll_fault_idx, ll_comp_cav):
        """Create the Faults."""
        l_faults_obj = []
        # Unpack the list of list of faulty indexes
        for l_idx, l_comp_cav in zip(ll_fault_idx, ll_comp_cav):
            l_faulty_cav = [self.brok_lin.elts[idx] for idx in l_idx]
            set_nature = {cav.get('nature', to_numpy=False)
                          for cav in l_faulty_cav}

            if not set_nature == {"FIELD_MAP"}:
                logging.error("At least one required element is not a "
                              + "FIELD_MAP.")
                raise IOError("At least one required element is not a "
                              + "FIELD_MAP.")

            # Create Fault object and append it to the list of Fault objects
            new_fault = mod_f.Fault(self.ref_lin, self.brok_lin, l_faulty_cav,
                                    l_comp_cav, self.wtf)
            l_faults_obj.append(new_fault)

        return l_faults_obj

    def _update_status_of_cavities_to_rephase(self):
        """
        Change the status of some cavities to 'rephased'.

        If the calculation is in relative phase, all cavities that are after
        the first failed one are rephased.
        """
        logging.warning(
            "The phases in the broken linac are relative. It may be more "
            + "relatable to use absolute phases, as it would avoid the "
            + "rephasing of the linac at each cavity.")
        idx_first_failed = self.faults['l_idx'][0][0]

        to_rephase_cavities = [
            cav for cav in self.brok_lin.elts[idx_first_failed:]
            if cav.get('status') == 'nominal'
            and cav.get('nature') == 'FIELD_MAP']
        # (the status of non-cavities is None, so they would be implicitely
        # excluded even without the nature checking)
        for cav in to_rephase_cavities:
            cav.update_status('rephased (in progress)')

    def _transfer_phi0_from_ref_to_broken(self):
        """
        Transfer the entry phases from ref linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when con.FLAG_PHI_ABS = True.
        """
        # Get CavitieS of REFerence and BROKen linacs
        ref_cs = self.ref_lin.l_cav#.get_elts('nature', 'FIELD_MAP')
        brok_cs = self.brok_lin.l_cav#get_elts('nature', 'FIELD_MAP')

        for ref_c, brok_c in zip(ref_cs, brok_cs):
            ref_a_f = ref_c.acc_field
            brok_a_f = brok_c.acc_field

            brok_a_f.phi_0['phi_0_abs'] = ref_a_f.phi_0['phi_0_abs']
            brok_a_f.phi_0['phi_0_rel'] = ref_a_f.phi_0['phi_0_rel']
            brok_a_f.phi_0['nominal_rel'] = ref_a_f.phi_0['phi_0_rel']

    def fix_all(self):
        """Fix the linac."""
        l_successes = []

        # We fix all Faults individually
        for i, fault in enumerate(self.faults['l_obj']):
            # If provided, we take the fault.info of another fault
            info_other_sol = None
            if self.l_info_other_sol is not None:
                info_other_sol = self.l_info_other_sol[i]

            success, d_sol = fault.fix(info_other_sol=info_other_sol)

            # Recompute transfer matrices to transfer proper rf_field, transfer
            # matrix, etc to _Elements, _Particles and _Accelerators.
            d_fits = {'l_phi': d_sol['X'][:fault.comp['n_cav']],
                      'l_k_e': d_sol['X'][fault.comp['n_cav']:],
                      'phi_s fit': self.wtf['phi_s fit']}

            results = fault.elts.compute_transfer_matrices(
                d_fits=d_fits, transfer_data=True)
            self.brok_lin.store_results(results, fault.elts)
            fault.get_x_sol_in_real_phase()

            # Update status of the compensating cavities according to the
            # success or not of the fit.
            fault.update_status(success)
            l_successes.append(success)
            self._compute_matrix_to_next_fault(fault, success)

            if not con.FLAG_PHI_ABS:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)

        # At the end we recompute the full transfer matrix
        # self.brok_lin.compute_transfer_matrices()
        results = self.brok_lin.elts.compute_transfer_matrices()
        results["mismatch factor"] = self._compute_mismatch()
        self.brok_lin.store_results(results, self.brok_lin.elts)
        self.brok_lin.name = f"Fixed ({str(l_successes.count(True))}" \
            + f" of {str(len(l_successes))})"

        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, mod_f.debugs['cav'])

        self.info['fit'] = debug.output_fit(self, mod_f.debugs['fit_complete'],
                                            mod_f.debugs['fit_compact'])

    def _compute_matrix_to_next_fault(self, fault, success):
        """Recompute transfer matrices between this fault and the next."""
        l_faults = self.faults['l_obj']
        l_elts = self.brok_lin.elts
        idx1 = fault.elts[-1].idx['elt_idx']

        if fault is not l_faults[-1] and success:
            next_fault = l_faults[l_faults.index(fault) + 1]
            idx2 = next_fault.fail['l_cav'][0].idx['elt_idx'] + 1
        else:
            idx2 = len(l_elts)

        elt1_to_elt2 = l_elts[idx1:idx2]
        s_elt1 = l_elts[idx1].idx['s_in']
        w_kin = self.brok_lin.get('w_kin')[s_elt1]
        phi_abs = self.brok_lin.get('phi_abs_array')[s_elt1]
        transf_mat = self.brok_lin.transf_mat['tm_cumul'][s_elt1]

        elts = ListOfElements(elt1_to_elt2, w_kin=w_kin, phi_abs=phi_abs,
                              idx_in=s_elt1, tm_cumul=transf_mat)
        results = elts.compute_transfer_matrices()
        self.brok_lin.store_results(results, elts)

    def _reupdate_status_of_rephased_cavities(self, fault):
        """
        Modify the status of the cavities that were already rephased.

        Change the cavities with status "rephased (in progress)" to
        "rephased (ok)" between this fault and the next one.
        """
        logging.warning("Changed the way of defining idx1 and idx2.")
        l_faults = self.faults['l_obj']
        l_elts = self.brok_lin.elts

        # idx1 = l_elts.index(fault.comp['l_all_elts'][-1])
        idx1 = fault.elts[-1].idx['elt_idx']
        if fault is l_faults[-1]:
            idx2 = len(l_elts)
        else:
            next_fault = l_faults[l_faults.index(fault) + 1]
            # idx2 = l_elts.index(next_fault.comp['l_all_elts'][0]) + 1
            idx2 = next_fault.elts[0].idx('elt_idx') + 1

        l_cav_between_two_faults = [elt for elt in l_elts[idx1:idx2]
                                    if elt.get('nature') == 'FIELD_MAP']
        for cav in l_cav_between_two_faults:
            if cav.get('status') == 'rephased (in progress)':
                cav.update_status('rephased (ok)')

    def evaluate_fit_quality(self, delta_t, user_idx=None):
        """Compute some quantities on the whole linac to see if fit is good."""
        keys = ['w_kin', 'phi_abs_array', 'envelope_pos_w',
                'envelope_energy_w', 'mismatch factor', 'eps_w']
        val = {}
        for key in keys:
            val[key] = []

        # End of each compensation zone
        l_idx = [fault.elts[-1].get('s_out') for fault in self.faults['l_obj']]
        str_columns = [f"end comp zone\n(idx {idx}) [%]"
                       for idx in l_idx]

        # If user provided more idx to check
        if user_idx is not None:
            l_idx += user_idx
            str_columns += [f"user defined\n(idx {idx}) [%]"
                            for idx in user_idx]

        # End of linac
        l_idx.append(-1)
        str_columns.append("end linac [%]")

        # First column labels
        str_columns.insert(0, "Qty")

        # Calculate relative errors in %
        for idx in l_idx:
            for key in keys:
                ref = self.ref_lin.get(key)[idx]
                fix = self.brok_lin.get(key)[idx]

                if key == 'mismatch factor':
                    val[key].append(fix)
                    continue
                val[key].append(1e2 * (ref - fix) / ref)

        # Relative square difference sumed on whole linac
        str_columns.append("sum error linac")
        for key in keys:
            ref = self.ref_lin.get(key)
            ref[ref == 0.] = np.NaN
            fix = self.brok_lin.get(key)

            if key == 'mismatch factor':
                val[key].append(np.sum(fix))
                continue
            val[key].append(np.nansum(np.sqrt(((ref - fix) / ref)**2)))

        # Handle time
        time_line = [None for n in range(len(val[keys[0]]))]
        days, seconds = delta_t.days, delta_t.seconds
        time_line[0] = f"{days * 24 + seconds // 3600} hrs"
        time_line[1] = f"{seconds % 3600 // 60} min"
        time_line[2] = f"{seconds % 60} sec"

        # Now make it a pandas dataframe for sweet output
        df_eval = pd.DataFrame(columns=str_columns)
        df_eval.loc[0] = ['time'] + time_line
        for i, key in enumerate(keys):
            df_eval.loc[i + 1] = [key] + val[key]

        return df_eval

    def _compute_mismatch(self) -> np.ndarray:
        """
        Compute the mismatch between reference abnd broken linac.

        Also store it into the broken_linac.beam_param dictionary.
        """
        mism = mismatch_factor(self.ref_lin.get("twiss_z"),
                               self.brok_lin.get("twiss_z"), transp=True)
        return mism


def neighboring_cavities(lin, l_faulty_cav, n_comp_per_fault):
    """Select the cavities neighboring the failed one(s)."""
    # l_all_cav = lin.get_elts('nature', 'FIELD_MAP')
    l_idx_faults = [lin.l_cav.index(faulty_cav)
                    for faulty_cav in l_faulty_cav]
    distances = []
    for idx_f in l_idx_faults:
        # Distance between every cavity and the cavity under study
        distance = np.array([idx_f - lin.l_cav.index(cav)
                             for cav in lin.l_cav], dtype=np.float64)
        distances.append(np.abs(distance))

    # Distance between every cavity and it's closest fault
    distances = np.array(distances)
    distance = np.min(distances, axis=0)
    n_cav_to_take = len(l_idx_faults) * (1 + n_comp_per_fault)

    # To favorise cavities near the start of the linac when there is an
    # equality in distance
    sort_bis = np.linspace(1, len(lin.l_cav), len(lin.l_cav))

    # To favorise the cavities near the end of the linac, just invert this
    # sort_bis = -sort_bis

    idx_compensating = np.lexsort((sort_bis, distance))[:n_cav_to_take]
    idx_compensating.sort()
    l_comp_cav = [lin.l_cav[idx] for idx in idx_compensating]
    return l_comp_cav


def neighboring_lattices(lin, l_faulty_cav, n_lattices_per_fault):
    """Select full lattices neighboring the failed cavities."""
    assert n_lattices_per_fault % 2 == 0, "Need an even number of" \
        + " compensating lattices per fault."
    # List of all lattices
    l_all_latt = [lattice
                  for section in lin.elements['l_sections']
                  for lattice in section]
    # List of lattices with a fault
    l_faulty_latt = [lattice
                     for faulty_cav in l_faulty_cav
                     for lattice in l_all_latt
                     if faulty_cav in lattice]
    # Index of these faulty lattices
    l_idx_faulty_latt = [l_all_latt.index(faulty_lattice)
                         for faulty_lattice in l_faulty_latt]

    half_n_latt = int(len(l_faulty_cav) * n_lattices_per_fault / 2)
    # List of compensating lattices
    l_comp_latt = l_all_latt[l_idx_faulty_latt[0] - half_n_latt:
                             l_idx_faulty_latt[-1] + half_n_latt + 1]

    # List of cavities in the compensating lattices
    l_comp_cav = [element
                  for lattice in l_comp_latt
                  for element in lattice
                  if element.get('nature') == 'FIELD_MAP']
    return l_comp_cav


def manually_set_cavities(lin, l_faulty_idx, l_comp_idx):
    """Select cavities that were manually defined."""
    types = {type(x) for x in l_faulty_idx + l_comp_idx}
    assert types == {list}, "Need a list of lists of indexes."

    assert len(l_faulty_idx) == len(l_comp_idx), "Need a list of compensating"\
        + " cavities index for each list of faults."

    natures = {lin.elts[idx].get('nature')
               for l_idx in l_faulty_idx + l_comp_idx
               for idx in l_idx}
    assert natures == {'FIELD_MAP'}, "All faulty and compensating elements" \
        + " must be 'FIELD_MAP's."

    l_comp_cav = [[lin.elts[idx]
                   for idx in l_idx]
                  for l_idx in l_comp_idx]
    return l_comp_cav


def all_cavities_after_first_fault(lin, l_faulty_cav):
    """Return all the cavities after the failure."""
    l_elts = lin.elts
    l_idx_faults = [l_elts.index(cav) for cav in l_faulty_cav]
    first = min(l_idx_faults)
    l_elts = l_elts[first:]
    l_comp_cav = [cav for cav in l_elts if cav.get('nature') == 'FIELD_MAP'
                  and cav not in l_faulty_cav]
    return l_comp_cav
