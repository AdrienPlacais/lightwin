#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022.

@author: placais

Module holding the FaultScenario, which holds the Faults. Each Fault object
fixes himself (Fault.fix_single), and a second optimization is performed to
smoothen the individual fixes. # TODO

brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.

TODO remake a small fit after the first one?
TODO plot interesting data before the second fit to see if it is
useful
"""
import itertools
import math
from constants import FLAG_PHI_ABS, WHAT_TO_FIT
from helper import printc
import debug
import fault as mod_f


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac, l_fault_idx):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        assert ref_linac.synch.info['reference'] is True
        assert broken_linac.synch.info['reference'] is False

        l_fault_idx = sorted(l_fault_idx)
        l_obj, l_comp_cav, l_fault_idx = \
            self._gather_and_create_fault_objects(l_fault_idx)

        self.faults = {'l_obj': l_obj,          # List of Fault objects
                       # List of list of index of failed cavities, grouped by
                       # Fault
                       'l_idx': l_fault_idx,
                       # List of list of compensating + failed cavities,
                       # grouped by Fault
                       'l_comp': l_comp_cav,
        }
        # FIXME move elsewhere
        # Ensure that the cavities are sorted from linac entrance to linac exit
        flattened_l_fault_idx = [idx for l_idx in l_fault_idx for idx in l_idx]
        assert flattened_l_fault_idx == sorted(flattened_l_fault_idx)

        self.info = {'fit': None}

        # Change status of cavities after the first failed cavity to tell LW
        # that they must keep their relative entry phases, not their absolute
        if not FLAG_PHI_ABS:
            self._update_status_of_cavities_to_rephase()

    def transfer_phi0_from_ref_to_broken(self):
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

            brok_a_f.phi_0['abs'] = ref_a_f.phi_0['abs']
            brok_a_f.phi_0['rel'] = ref_a_f.phi_0['rel']
            brok_a_f.phi_0['nominal_rel'] = ref_a_f.phi_0['rel']

    def fix_all(self):
        """
        Fix the linac.

        First, fix all the Faults independently. Then, recompute the linac
        and make small adjustments.
        """
        self._prepare_compensating_cavities_of_all_faults()

        l_flags_success = []

        # We fix all Faults individually
        for fault in self.faults['l_obj']:
            flag_success, opti_sol = fault.fix_single()

            # Recompute transfer matrices to transfer proper rf_field, transfer
            # matrix, etc to _Elements, _Particles and _Accelerators.
            d_fits = {'l_phi': opti_sol[:fault.comp['n_cav']].tolist(),
                      'l_k_e': opti_sol[fault.comp['n_cav']:].tolist()}
            fault.brok_lin.compute_transfer_matrices(fault.comp['l_recompute'],
                                                     d_fits=d_fits,
                                                     flag_transfer_data=True)

            # Update status of the compensating cavities according to the
            # success or not of the fit.
            fault.update_status(flag_success)
            l_flags_success.append(flag_success)
            self._compute_matrix_to_next_fault(fault, flag_success)

            if not FLAG_PHI_ABS:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)

        # At the end we recompute the full transfer matrix
        self.brok_lin.compute_transfer_matrices()
        self.brok_lin.name = f"Fixed ({str(l_flags_success.count(True))}" \
            + f" of {str(len(l_flags_success))})"

        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, mod_f.debugs['cav'])

        self.info['fit'] = debug.output_fit(self, mod_f.debugs['fit_complete'],
                                            mod_f.debugs['fit_compact'])

    def _compute_matrix_to_next_fault(self, fault, success):
        """Recompute transfer matrices between this fault and the next."""
        l_faults = self.faults['l_obj']
        l_elts = self.brok_lin.elements['list']
        idx1 = l_elts.index(fault.comp['l_all_elts'][-1])

        if fault is not l_faults[-1] and success:
            next_fault = l_faults[l_faults.index(fault) + 1]
            idx2 = l_elts.index(next_fault.comp['l_all_elts'][0]) + 1
            print('fault_scenario: recompute only until index', idx2)
        else:
            idx2 = len(l_elts)
            print('fault_scenario: recompute until end')

        elt1_to_elt2 = l_elts[idx1:idx2]
        self.brok_lin.compute_transfer_matrices(elt1_to_elt2)

    def _gather_and_create_fault_objects(self, l_fault_idx):
        """Gather faults that are close to each other, create Faults."""
        lin = self.brok_lin
        l_faulty_cav = [lin.elements['list'][idx]
                        for idx in l_fault_idx]
        assert all(elem.info['nature'] == 'FIELD_MAP'
                   for elem in l_faulty_cav), \
            'Not all failed cavities that you asked are cavities.'

        d_comp_cav = {
            'k out of n': lambda l_cav:
                neighboring_cavities(lin, l_cav, WHAT_TO_FIT['k']),
            'l neighboring lattices': lambda l_cav:
                neighboring_lattices(lin, l_cav, WHAT_TO_FIT['l']),
        }

        # List of list of faults indexes
        l_gathered_idx_faults = [[idx] for idx in l_fault_idx]
        # List of list of corresp. faulty cavities
        l_gathered_faults = [[lin.elements['list'][idx]
                              for idx in l_idx]
                             for l_idx in l_gathered_idx_faults]
        flag_gathered = False
        r_comb = 2

        while not flag_gathered:
            # List of list of corresp. compensating cavities
            l_gathered_comp = [d_comp_cav[WHAT_TO_FIT['strategy']](l_cav)
                               for l_cav in l_gathered_faults]

            # Set a counter to exit the 'for' loop when all faults are gathered
            i = 0
            n_comb = len(l_gathered_comp)
            if n_comb <= 1:
                flag_gathered = True

            i_max = int(math.factorial(n_comb) / (
                math.factorial(r_comb) * math.factorial(n_comb - r_comb)))

            # Now we look every list of required compensating cavities, and
            # look for faults that require the same compensating cavities
            for ((idx1, l_comp1), (idx2, l_comp2)) \
                in itertools.combinations(enumerate(l_gathered_comp),
                                          r_comb):
                i += 1
                common_cav = list(set(l_comp1) & set(l_comp2))
                # If at least one cavity on common, gather the two
                # corresponding fault and restart the whole process
                if len(common_cav) > 0:
                    l_gathered_faults[idx1].extend(
                        l_gathered_faults.pop(idx2))
                    l_gathered_idx_faults[idx1].extend(
                        l_gathered_idx_faults.pop(idx2))
                    break

                # If we reached this point, it means that there is no list of
                # faults that share compensating cavities.
                if i == i_max:
                    flag_gathered = True

        l_faults_obj = []
        for f_cav in l_gathered_faults:
            l_faults_obj.append(
                mod_f.Fault(self.ref_lin, self.brok_lin, f_cav)
            )
        return l_faults_obj, l_gathered_comp, l_gathered_idx_faults

    def _prepare_compensating_cavities_of_all_faults(self):
        """Call fault.prepare_cavities_for_compensation."""
        # FIXME should be moved into the function that regroups the faults
        if WHAT_TO_FIT['strategy'] == 'manual':
            l_comp_idx = WHAT_TO_FIT['manual list']
            assert len(l_comp_idx) == len(self.faults['l_obj']), "There" \
                    + " should be a list of compensating cavities for every" \
                    + " fault."
            l_comp_cav = [[self.brok_lin.elements['list'][idx]
                           for idx in l_idx]
                          for l_idx in l_comp_idx]
        else:
            l_comp_cav = self.faults['l_comp']

        for fault, comp_cav in zip(self.faults['l_obj'], l_comp_cav):
            fault.prepare_cavities_for_compensation(comp_cav)

    def _update_status_of_cavities_to_rephase(self):
        """
        Change the status of some cavities to 'rephased'.

        If the calculation is in relative phase, all cavities that are after
        the first failed one are rephased.
        """
        printc("fault_scenario._update_status_of_cavities_to_rephase warning:",
               opt_message = " the phases in the broken linac are relative." \
               + " It may be more relatable to use absolute phases, as" \
               + " it would avoid the rephasing of the linac at each cavity.")
        all_elements = self.brok_lin.elements['list']
        idx_first_failed = self.faults['l_idx'][0][0]

        to_rephase_cavities = [cav for cav in all_elements[idx_first_failed:]
                               if cav.info['status'] == 'nominal']
        # The status of non-cavities is None, so they are implicitely excluded
        for cav in to_rephase_cavities:
            cav.update_status('rephased (in progress)')

    def _reupdate_status_of_rephased_cavities(self, fault):
        """
        Modify the status of the cavities that were already rephased.

        Change the cavities with status "rephased (in progress)" to
        "rephased (ok)" between this fault and the next one.
        """
        l_faults = self.faults['l_obj']
        l_elts = self.brok_lin.elements['list']

        idx1 = l_elts.index(fault.comp['l_all_elts'][-1])
        if fault is l_faults[-1]:
            idx2 = len(l_elts)
        else:
            next_fault = l_faults[l_faults.index(fault) + 1]
            idx2 = l_elts.index(next_fault.comp['l_all_elts'][0]) + 1

        l_cav_between_two_faults = [elt for elt in l_elts[idx1:idx2]
                                    if elt.info['nature'] == 'FIELD_MAP']
        for cav in l_cav_between_two_faults:
            if cav.info['status'] == 'rephased (in progress)':
                cav.update_status('rephased (ok)')


def neighboring_cavities(lin, l_faulty_cav, n_comp_per_fault):
    """Select the cavities neighboring the failed one(s). """
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
                  if element.info['nature'] == 'FIELD_MAP']
    return l_comp_cav
