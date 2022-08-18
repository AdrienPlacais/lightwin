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
"""
import itertools
from constants import FLAG_PHI_ABS, WHAT_TO_FIT
import debug
import fault as mod_f


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac, l_fault_idx, l_comp_idx):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        assert ref_linac.synch.info['reference'] is True
        assert broken_linac.synch.info['reference'] is False

        # Save faults as a list of Fault objects and as a list of cavity idx
        l_fault_idx = sorted(l_fault_idx)
        self.faults = {
            'l_obj': self._gather_and_create_fault_objects(l_fault_idx),
            'l_idx': l_fault_idx
        }

        self.info = {'fit': None}

        self._update_status_of_cavities_that_compensate(l_comp_idx)
        # If the calculation is made in relative phase, we change the status
        # of all the cavities after the first fault to tell LightWin that it
        # must conserve the cavities RELATIVE phi_0, not the absolute one.
        # Set compensating cavities
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
        l_flags_success = []
        # We fix all Faults individually
        for fault in self.faults['l_obj']:
            flag_success, opti_sol = fault.fix_single()
            l_flags_success.append(flag_success)

            # The norms and phi_0 from sol.x will be transfered to the electric
            # field objects thanks to transfer_data=True
            d_fits = {'flag': True,
                      'l_phi': opti_sol[:fault.comp['n_cav']].tolist(),
                      'l_norm': opti_sol[fault.comp['n_cav']:].tolist(),
                     }

            # Recompute transfer matrix with proper solution
            self.brok_lin.compute_transfer_matrices(
                fault.comp['l_recompute'], d_fits=d_fits,
                flag_transfer_data=True)

            # Recompute transfer matrix between the end of this compensating
            # zone and the start of the next (or to the linac end)
            self._compute_matrix_to_next_fault(fault, flag_success)

            if not FLAG_PHI_ABS:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)

        # TODO plot interesting data before the second fit to see if it is
        # useful
        # TODO remake a small fit to be sure

        # At the end we recompute the full transfer matrix
        self.brok_lin.compute_transfer_matrices()
        self.brok_lin.name = f"Fixed ({str(l_flags_success.count(True))} of {str(len(l_flags_success))})"

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
        self.brok_lin.compute_transfer_matrices(elt1_to_elt2,
                                                flag_transfer_data=True)

    def _gather_and_create_fault_objects(self, l_fault_idx):
        """
        Gather faults in the same or neighboring lattices, create Faults.

        Parameters
        ----------
        l_fault_idx : list
            List of the indices (in linac.elements['list']) of the faulty
            linacs.

        Return
        ------
        l_faults_obj : list of Faults
            List of the Fault objects.
        """
        assert mod_f.N_COMP_LATT_PER_FAULT % 2 == 0, \
            'You need an even number of compensating lattices per faulty '\
            + 'cav to distribute them equally.'
        assert all([
            self.brok_lin.elements['list'][idx].info['nature'] == 'FIELD_MAP'
            for idx in l_fault_idx
        ]), 'Not all failed cavities that you asked are cavities.'

        def are_close(idx1, idx2):
            latt1 = self.brok_lin.elements['list'][idx1].idx['lattice'][0]
            latt2 = self.brok_lin.elements['list'][idx2].idx['lattice'][0]
            return abs(latt1 - latt2) <= mod_f.N_COMP_LATT_PER_FAULT / 2
        # Regroup faults that are too close to each other as they will be fixed
        # at the same moment
        grouped_faults_idx = [[idx1
                               for idx1 in l_fault_idx
                               if are_close(idx1, idx2)]
                              for idx2 in l_fault_idx]
        # Remove doublons
        grouped_faults_idx = list(grouped_faults_idx
                                  for grouped_faults_idx, _
                                  in itertools.groupby(grouped_faults_idx))

        # Create Fault objects
        l_faults_obj = []
        for f_idx in grouped_faults_idx:
            l_faults_obj.append(
                mod_f.Fault(self.ref_lin, self.brok_lin, f_idx)
            )
        return l_faults_obj

    def _update_status_of_cavities_that_compensate(self, l_comp_idx):
        """Call set_compensating_cavities froms faults with proper args."""
        if WHAT_TO_FIT['strategy'] == 'manual':
            msg = "There should be a list of compensating cavities for" \
                    + "every fault."
            assert len(l_comp_idx) == len(self.faults['l_obj']), msg

        for fault_idx, fault_obj in enumerate(self.faults['l_obj']):
            try:
                sub_l_comp_idx = l_comp_idx[fault_idx]
            except IndexError:
                sub_l_comp_idx = None

            fault_obj.set_compensating_cavities(WHAT_TO_FIT['strategy'],
                                                sub_l_comp_idx)

    def _update_status_of_cavities_to_rephase(self):
        """
        Change the status of some cavities to 'rephased'.

        If the calculation is in relative phase, all cavities that are after
        the first failed one are rephased.
        ---
        Legacy, I do not know why I wrote this:
        ---
        Even in the case of an absolute phase calculation, cavities in the
        HEBT are rephased.
        """
        print("Warning, the phases in the broken linac are relative.",
              "It may be more relatable to use absolute phases, as",
              "it would avoid the rephasing of the linac at each cavity.")

        # We get first failed cav index
        ffc_idx = min(self.faults['l_fault_idx'])
        after_ffc = self.brok_lin.elements['list'][ffc_idx:]

        cav_to_rephase = [
            cav for cav in after_ffc
            if (cav.info['nature'] == 'FIELD_MAP'
                and cav.info['status'] == 'nominal')
            # and (cav.info['zone'] == 'HEBT' or not FLAG_PHI_ABS)
        ]
        for cav in cav_to_rephase:
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
