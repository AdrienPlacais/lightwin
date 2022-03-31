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
from constants import FLAG_PHI_ABS
import debug
import fault as mod_f


class FaultScenario():
    """A class to hold all fault related data."""

    def __init__(self, ref_linac, broken_linac, what_to_fit, l_idx_fault,
                 l_idx_comp):
        self.ref_lin = ref_linac
        self.brok_lin = broken_linac

        assert ref_linac.synch.info['reference'] is True
        assert broken_linac.synch.info['reference'] is False
        self.transfer_phi0_from_ref_to_broken()

        self.what_to_fit = {
            # How are selected the compensating cavities?
            'strategy': what_to_fit['strategy'],
            # What do we want to fit?
            'objective': what_to_fit['objective'],
            # Where are we measuring 'objective'?
            'position': what_to_fit['position'],
            'fit_over_phi_s': what_to_fit['fit_over_phi_s'],
            }

        # Save faults as a list of Fault objects and as a list of cavity idx
        l_idx_fault = sorted(l_idx_fault)
        self.faults = {
            'l_obj': self._distribute_and_create_fault_objects(
                l_idx_fault, l_idx_comp),
            'l_idx': l_idx_fault,
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

    def _distribute_and_create_fault_objects(self, l_idx_fault, l_idx_comp):
        """
        Create the Fault objects.

        First we gather faults indices that are in the same or neighboring
        lattices.
        Then we determine which cavities will compensate these faults. If two
        different faults need the same cavity, we merge them.

        Parameters
        ----------
        l_idx_fault : list
            List of the indices (in linac.elements['list']) of the faulty
            linacs.
        l_idx_comp : list of lists
            List of list of indices (same) of the cavity that will be used to
            compensate the faults. The len of this list should match the number
            of faults.
        """
        assert mod_f.n_comp_latt_per_fault % 2 == 0, \
            'You need an even number of compensating lattices per faulty '\
            + 'cav to distribute them equally.'
        assert all([
            self.brok_lin.elements['list'][idx].info['nature'] == 'FIELD_MAP'
            for idx in l_idx_fault
            ]), 'Not all failed cavities that you asked are cavities.'

        def are_close(idx1, idx2):
            latt1 = self.brok_lin.elements['list'][idx1].idx['lattice'][0]
            latt2 = self.brok_lin.elements['list'][idx2].idx['lattice'][0]
            return abs(latt1 - latt2) <= mod_f.n_comp_latt_per_fault / 2
        # Regroup faults that are too close to each other as they will be fixed
        # at the same moment
        grouped_faults_idx = [[idx1
                               for idx1 in l_idx_fault
                               if are_close(idx1, idx2)]
                              for idx2 in l_idx_fault]
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
        if self.what_to_fit['strategy'] == 'neighbors':
            for fault in l_faults_obj:
                all_comp_cav.append(fault.select_neighboring_cavities())
            print('Check that there is no cavity in common.')

        elif self.what_to_fit['strategy'] == 'manual':
            assert len(l_idx_comp) == len(l_faults_obj), \
                'Error, the number of group of compensating cavities do not' \
                + ' match the number of Fault objects.'
            all_comp_cav = [
                [self.brok_lin.elements['list'][idx]
                 for idx in idx_list]
                for idx_list in l_idx_comp
                ]

        # Change status (failed, compensate) of the proper cavities, create the
        # fault.comp['l_all_elts'] list containing the full lattices in which
        # we have our compensating and faulty cavities
        for (f, l_comp_cav) in zip(l_faults_obj, all_comp_cav):
            f.prepare_cavities(l_comp_cav)

        return l_faults_obj

    def fix_all(self, method):
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
            suc = f.fix_single(method, self.what_to_fit)
            successes.append(suc)

            # Recompute transfer matrices between this fault and the next
            if i < len(self.faults['l_obj']) - 1 and suc:
                print('fix_all: computing mt between two errors...')
                # FIXME: necessary to take '-2'? This is just to be sure...
                elt1 = f.comp['l_all_elts'][-2]
                idx1 = self.brok_lin.elements['list'].index(elt1)

                if i < len(self.faults['l_obj'] - 1):
                    elt2 = self.faults['l_obj'][i+1].comp['l_all_elts'][0]
                else:
                    elt2 = self.brok_lin.elements['list'][-1]
                idx2 = self.brok_lin.elements['list'].index(elt2)

                elt1_to_elt2 = self.brok_lin.elements['list'][idx1:idx2+1]
                self.brok_lin.compute_transfer_matrices(method, elt1_to_elt2)
        # TODO plot interesting data before the second fit to see if it is
        # useful
        # TODO we remake a small fit to be sure

        # At the end we recompute the full transfer matrix
        # self.brok_lin.compute_transfer_matrices(method, flag_synch=False)
        self.brok_lin.name = 'Fixed (' + str(successes.count(True)) + '/' + \
            str(len(successes)) + ')'

        for linac in [self.ref_lin, self.brok_lin]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, mod_f.debugs['cav'])
        self.info['fit'] = debug.output_fit(self, mod_f.debugs['fit_complete'],
                                            mod_f.debugs['fit_compact'])
