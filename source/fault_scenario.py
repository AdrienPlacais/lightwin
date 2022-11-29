#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022.

@author: placais

Module holding the FaultScenario, which holds the Faults. Each Fault object
fixes himself (Fault.fix), and a second optimization is performed to
smoothen the individual fixes. # TODO

brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.

TODO handle faults at linac extremities
TODO allow for different strategies according to the section
TODO raise explicit error when the format of error (list vs idx) is not
appropriate, especially when manual mode.
TODO allow for uneven distribution of compensating cavities (ex: k out of n=5)

TODO tune the PSO
TODO method to avoid big changes of acceptance
TODO option to minimize the power of compensating cavities

TODO remake a small fit after the first one?
TODO plot interesting data before the second fit to see if it is
useful
"""
import itertools
import math
import numpy as np
import pandas as pd
from constants import FLAG_PHI_ABS
from list_of_elements import ListOfElements
from helper import printc
import debug
import fault as mod_f


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac, l_fault_idx, wtf,
                 l_info_other_sol=None):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac
        self.wtf = wtf
        self.l_info_other_sol = l_info_other_sol

        assert ref_linac.get('reference') and not broken_linac.get('reference')

        ll_comp_cav, ll_fault_idx = self._sort_faults(l_fault_idx)
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
        if not FLAG_PHI_ABS:
            self._update_status_of_cavities_to_rephase()
        self._transfer_phi0_from_ref_to_broken()

        results = self.brok_lin.elts.compute_transfer_matrices()
        self.brok_lin.save_results(results, self.brok_lin.elts)
        self.brok_lin.compute_mismatch(self.ref_lin)

    def _sort_faults(self, l_fault_idx):
        """Gather faults that are close to each other."""
        lin = self.brok_lin

        # If in manual mode, faults should be already gathered
        if self.wtf['strategy'] == 'manual':
            l_faulty_cav = [[lin.elts[idx]
                             for idx in l_idx]
                            for l_idx in l_fault_idx]
            ll_idx_faults = l_fault_idx
            ll_comp = manually_set_cavities(lin, l_fault_idx,
                                            self.wtf['manual list'])

        else:
            l_faulty_cav = [lin.elts[idx]
                            for idx in sorted(l_fault_idx)]

            # Initialize list of list of faults indexes
            ll_idx_faults = [[idx] for idx in sorted(l_fault_idx)]
            # Initialize list of list of corresp. faulty cavities
            ll_faults = [[lin.elts[idx]
                          for idx in l_idx]
                         for l_idx in ll_idx_faults]

            # We go across all faults and determine the compensating cavities
            # they need. If two failed cavities need at least one compensating
            # cavity in common, we group them together.
            # In particular, it is the case when a full cryomodule fails.
            ll_comp, ll_idx_faults = self._gather(ll_faults, ll_idx_faults)

        return ll_comp, ll_idx_faults

    def _gather(self, ll_faults, ll_idx_faults):
        """Proper method that gathers faults requiring the same compens cav."""
        d_comp_cav = {
            'k out of n': lambda l_cav:
                neighboring_cavities(self.brok_lin, l_cav, self.wtf['k']),
            'l neighboring lattices': lambda l_cav:
                neighboring_lattices(self.brok_lin, l_cav, self.wtf['l'])}
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
            # Get faulty cavities
            l_faulty_cav = [self.brok_lin.elts[idx] for idx in l_idx]

            # Check that they are all cavities
            set_nature = {cav.get('nature', to_numpy=False)
                          for cav in l_faulty_cav}
            assert set_nature == {"FIELD_MAP"}

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
        printc("fault_scenario._update_status_of_cavities_to_rephase warning:",
               opt_message=" the phases in the broken linac are relative."
               + " It may be more relatable to use absolute phases, as"
               + " it would avoid the rephasing of the linac at each cavity.")
        idx_first_failed = self.faults['l_idx'][0][0]

        to_rephase_cavities = [cav for cav in self.brok_lin.elts[idx_first_failed:]
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
        want to avoid when FLAG_PHI_ABS = True.
        """
        # Get CavitieS of REFerence and BROKen linacs
        ref_cs = self.ref_lin.elements_of('FIELD_MAP')
        brok_cs = self.brok_lin.elements_of('FIELD_MAP')

        for ref_c, brok_c in zip(ref_cs, brok_cs):
            ref_a_f = ref_c.acc_field
            brok_a_f = brok_c.acc_field

            brok_a_f.phi_0['phi_0_abs'] = ref_a_f.phi_0['phi_0_abs']
            brok_a_f.phi_0['phi_0_rel'] = ref_a_f.phi_0['phi_0_rel']
            brok_a_f.phi_0['nominal_rel'] = ref_a_f.phi_0['phi_0_rel']

    def fix_all(self):
        """
        Fix the linac.

        First, fix all the Faults independently. Then, recompute the linac
        and make small adjustments.
        """
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
            self.ref_lin.save_results(results, fault.elts)
            fault.get_x_sol_in_real_phase()

            # Update status of the compensating cavities according to the
            # success or not of the fit.
            fault.update_status(success)
            l_successes.append(success)
            self._compute_matrix_to_next_fault(fault, success)

            if not FLAG_PHI_ABS:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)

        # At the end we recompute the full transfer matrix
        # self.brok_lin.compute_transfer_matrices()
        results = self.brok_lin.elts.compute_transfer_matrices()
        self.brok_lin.save_results(results, self.brok_lin.elts)
        self.brok_lin.compute_mismatch(self.ref_lin)
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
            idx2 = next_fault.elts[0].idx('elt_idx') + 1
        else:
            idx2 = len(l_elts)

        elt1_to_elt2 = l_elts[idx1:idx2]
        s_elt1 = l_elts[idx1].idx['s_in']
        w_kin = self.brok_lin.get('w_kin')[s_elt1]
        phi_abs = self.brok_lin.get('phi_abs_array')[s_elt1]
        transf_mat = self.brok_lin.transf_mat['tm_cumul'][s_elt1]

        elts = ListOfElements(elt1_to_elt2, w_kin=w_kin, phi_abs=phi_abs,
                              idx_in=s_elt1, r_zz_cumul=transf_mat)
        results = elts.compute_transfer_matrices()
        self.brok_lin.save_results(results, elts)

    def _reupdate_status_of_rephased_cavities(self, fault):
        """
        Modify the status of the cavities that were already rephased.

        Change the cavities with status "rephased (in progress)" to
        "rephased (ok)" between this fault and the next one.
        """
        printc("fault_scenario._reupdate_status_of_rephased_cavities warning:",
               opt_message=" changed the way of defining idx1 and idx2.")
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

    # FIXME not Pythonic at all
    def evaluate_fit_quality(self, delta_t):
        """Compute some quantities on the whole linac to see if fit is good."""
        d_get = {
            'W_kin': lambda lin: lin.synch.energy['w_kin'],
            'phi': lambda lin: lin.get('phi_abs_array'),
            'sigma_phi': lambda lin: lin.get('envelopes_w')[:, 0],
            'sigma_w': lambda lin: lin.get('envelopes_w')[:, 1],
            'mismatch factor': lambda lin: lin.get('mismatch factor'),
            'emittance': lambda lin: lin.get('eps_w'),
        }
        idx_end_comp_zone = 824
        printc("Warning fault_scenario.evaluate_fit_quality: ",
               opt_message="Index of end compensation zone manually set.")

        def rel_sq_diff(ref, fix):
            delta = np.sqrt(((ref - fix) / ref)**2)
            delta_sum = np.nansum(delta)
            return delta_sum

        def rel_diff_end_linac(ref, fix):
            err = 1e2 * (ref[-1] - fix[-1]) / ref[-1]
            return err

        def rel_diff_end_comp_zone(ref, fix):
            err = 1e2 * (ref[idx_end_comp_zone] - fix[idx_end_comp_zone]) \
                / ref[idx_end_comp_zone]
            return err

        l_str_fun = ['end of comp. zone [%]', 'end of linac [%]',
                     'sum error on linac']
        df_ranking = pd.DataFrame(columns=(['Qty'] + l_str_fun))
        df_ranking.loc[0] = ['time', delta_t, None, None]

        criterions = d_get.keys()
        for i, crit in enumerate(criterions):
            l_rank = [crit]
            for str_fun in l_str_fun:
                if str_fun == 'end of linac [%]':
                    fun1 = rel_diff_end_linac
                    fun2 = lambda r_l, b_l: b_l[-1]
                elif str_fun == 'end of comp. zone [%]':
                    fun1 = rel_diff_end_comp_zone
                    fun2 = lambda r_l, b_l: b_l[idx_end_comp_zone]
                elif str_fun == 'sum error on linac':
                    fun1 = rel_sq_diff
                    fun2 = lambda r_l, b_l: np.nansum(b_l)
                else:
                    raise IOError

                d_delta = {
                    'W_kin': fun1,
                    'phi': fun1,
                    'sigma_phi': fun1,
                    'sigma_w': fun1,
                    'mismatch factor': fun2,
                    'emittance': fun1,
                }

                args = (d_get[crit](self.ref_lin), d_get[crit](self.brok_lin))
                delta = d_delta[crit](*args)
                l_rank.append(delta)
            df_ranking.loc[i + 1] = l_rank
        return df_ranking


def neighboring_cavities(lin, l_faulty_cav, n_comp_per_fault):
    """Select the cavities neighboring the failed one(s)."""
    assert n_comp_per_fault % 2 == 0, "Need an even number of compensating" \
        + " cavities per fault."
    l_all_cav = lin.elements_of(nature='FIELD_MAP')
    l_idx_faults = [l_all_cav.index(faulty_cav)
                    for faulty_cav in l_faulty_cav]

    n_faults = len(l_faulty_cav)
    half_n_comp = int(n_faults * n_comp_per_fault / 2)
    l_comp_cav = l_all_cav[l_idx_faults[0] - half_n_comp:
                           l_idx_faults[-1] + half_n_comp + 1]

    if len(l_comp_cav) > n_faults * (n_comp_per_fault + 1):
        printc("fault._select_neighboring_cavities warning: ",
               opt_message="the faults are probably not contiguous."
               + " Cavities between faulty cavities will be used for"
               + " compensation, thus increasing the number of"
               + " compensating cavites per fault.")
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
