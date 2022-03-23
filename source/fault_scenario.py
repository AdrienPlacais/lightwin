#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022

@author: placais

Module holding all the Fault_Scenario, which distribute the Faults. Each Fault
object fixes himself (Fault.fix_single), and a second optimization is performed
to smoothen the individual fixes. # TODO
brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.
"""
import itertools
from constants import FLAG_PHI_ABS
import debug
import fault as mod_f


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac, l_idx_cav):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        assert ref_linac.synch.info['reference'] is True
        assert broken_linac.synch.info['reference'] is False

        self.transfer_phi0_from_ref_to_broken()

        self.faults = {
            'l_obj': [],
            'l_idx': sorted(l_idx_cav),
                }
        self.list_of_faults = []    # TODO: remove
        self.faults['l_obj'] = self.distribute_faults()

        self.what_to_fit = {
            'strategy': None,   # How are selected the compensating cavities?
            'objective': None,  # What do we want to fit?
            'position': None,   # Where are we measuring 'objective'?
            }

        self.info = {'fit': None}

        if not FLAG_PHI_ABS:
            print('Warning, the phases in the broken linac are relative.',
                  'It may be more relatable to use absolute phases, as',
                  'it would avoid the implicit rephasing of the linac at',
                  'each cavity.\n')

    def transfer_phi0_from_ref_to_broken(self):
        """
        Transfer the entry phases from ref linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when FLAG_PHI_ABS = True.
        """
        ref_cavities = self.ref_lin.elements_of('FIELD_MAP')
        brok_cavities = self.brok_lin.elements_of('FIELD_MAP')
        assert len(ref_cavities) == len(brok_cavities)

        # Transfer both relative and absolute phase flags
        for i, ref_cav in enumerate(ref_cavities):
            ref_acc_f = ref_cav.acc_field
            brok_acc_f = brok_cavities[i].acc_field
            for str_phi_abs in ['rel', 'abs']:
                brok_acc_f.phi_0[str_phi_abs] = ref_acc_f.phi_0[str_phi_abs]
            brok_acc_f.phi_0['nominal_rel'] = ref_acc_f.phi_0['rel']

    def distribute_faults(self):
        """
        Create the Fault objects.

        First we gather faults indices that are in the same or neighboring
        lattices.
        Then we determine which cavities will compensate these faults. If two
        different faults need the same cavity, we merge them.
        """
        assert mod_f.n_comp_latt_per_fault % 2 == 0, \
            'You need an even number of compensating lattices per faulty '\
            + 'cav to distribute them equally.'
        assert all([
            self.brok_lin.elements['list'][idx].info['nature'] == 'FIELD_MAP'
            for idx in self.faults['l_idx']]), \
            'Not all failed cavities that you asked are cavities.'

        def are_close(idx1, idx2):
            latt1 = self.brok_lin.elements['list'][idx1].info['lattice_number']
            latt2 = self.brok_lin.elements['list'][idx2].info['lattice_number']
            return abs(latt1 - latt2) <= mod_f.n_comp_latt_per_fault / 2
        # Regroup faults that are too close to each other as they will be fixed
        # at the same moment
        grouped_faults_idx = [[idx1
                               for idx1 in self.faults['l_idx']
                               if are_close(idx1, idx2)]
                              for idx2 in self.faults['l_idx']]
        # Remove doublons
        grouped_faults_idx = \
            list(grouped_faults_idx
                 for grouped_faults_idx, _
                 in itertools.groupby(grouped_faults_idx))

        # Create Fault objects
        l_faults_obj = []
        for f_idx in grouped_faults_idx:
            l_faults_obj.append(
                mod_f.Fault(self.ref_lin, self.brok_lin, f_idx)
                )
        # Get cavities necessary for every Fault
        all_comp_cav = []
        for fault in l_faults_obj:
            all_comp_cav.append(fault.select_compensating_cavities())
        print('Check that there is no cavity in common.')

        for (fault, l_comp_cav) in zip(l_faults_obj, all_comp_cav):
            fault.update_status_cavities(l_comp_cav)

        return l_faults_obj

    def fix_all(self, method, what_to_fit, manual_list):
        """
        Fix the linac.

        First, fix all the Faults independently. Then, recompute the linac
        and make small adjustments.
        """
        # FIXME
        # self._select_cavities_to_rephase()

        # We fix all Faults individually
        successes = []
        for i, f in enumerate(self.faults['l_obj']):
            suc = f.fix_single(method, what_to_fit, manual_list)
            successes.append(suc)

            # Recompute transfer matrices between this fault and the next
            if i < len(self.faults['l_obj']) - 1:
                # FIXME: necessary to take '-2'? This is just to be sure...
                elt1 = f.comp['l_all_elts'][-2]
                elt2 = self.faults['l_obj'][i+1].comp['l_all_elts'][0]
                idx1 = self.brok_lin.elements['list'].index(elt1)
                idx2 = self.brok_lin.elements['list'].index(elt2)
                elt1_to_elt2 = self.brok_lin.elements['list'][idx1:idx2+1]
                self.brok_lin.compute_transfer_matrices(method, elt1_to_elt2)

        # TODO we remake a small fit to be sure
        # at the end we recompute the full transfer matrix
        self.brok_lin.compute_transfer_matrices(method)

        self.brok_lin.name = 'Fixed (' + str(successes.count(True)) + '/' + \
            str(len(successes)) + ')'

        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, mod_f.debugs['cav'])
        self.info['fit'] = debug.output_fit(self, out=True)
