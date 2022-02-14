#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022

@author: placais

Module holding all the fault-related functions.
brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.

blabla_idx is a list of indices. It is 'absolute', i.e. [5, 7, 15] refers to
the 5th, 7th and 15th elements of the linac, not to the 5th, 7th and 15th
cavities.
blabla_cav is a list of FIELD_MAP objects, matching blabla_idx.
"""


def apply_faults(brok_lin, fail_idx):
    """
    Break cavities at indices fail_idx.

    Parameters
    ----------
    brok_lin: Accelerator object
        Linac to break.
    fail_idx: list
        List of the position of the failing cavities.
    """
    brok_lin.fault_scenario['faults_idx'] = fail_idx

    for idx in fail_idx:
        cavity = brok_lin.list_of_elements[idx]
        assert cavity.name == 'FIELD_MAP', 'Error, the element at ' + \
            'position ' + str(idx) + ' is not a FIELD_MAP.'
        cavity.fail()
        brok_lin.fault_scenario['faults_cav'].append(cavity)


def compensate_faults(brok_lin, ref_lin, objective_str, strategy,
                      manual_list=None):
    """
    Compensate faults, according to strategy and objective.

    Parameters
    ----------
    brok_lin: Accelerator object
        Broken linac to fix.
    ref_lin: Accelerator object
        Reference linac, used for the fitting.
    objective_str: string
        Name of the attribute that shall correspond between brok_lin and
        ref_lin. Should be 'energy', 'phase' or 'transfer_matrix'.
    strategy: string
        Strategy that should be used for the choice of the compensating
        cavities.
        - 'neighbor': use neighbors of failed cavities. When two failed
        cavities already are next to each other, take neighbors of neighbors,
        so that we can have two compensating cavities per failing one.
        - 'all': use all cavities.
        - 'manual': use the list of indices manual_list.
    manual_list: list
        List of compensating cavities, used with strategy == 'manual'.
    """
    fault = brok_lin.fault_scenario

    # Select which cavities will be used to compensate the fault
    fault['strategy'] = strategy
    fault['objective_str'] = objective_str

    fault['comp_idx'], fault['comp_cav'] = \
        _select_compensating_cavities(brok_lin, strategy, manual_list)
    for elt in fault['comp_cav']:
        elt.status['compensate'] = True
        elt.acc_field.norm *= 1.9

    fault['objective'], idx_in, idx_out = \
        _select_objective(brok_lin, ref_lin, objective_str)

    # Loop over compensating cavities until objective is matched
    print('Warning, dirty.')
    comp_section = brok_lin.list_of_elements[manual_list[0]:manual_list[-1]+1]
    after_comp_section = brok_lin.list_of_elements[manual_list[-1]+1:]
    method = 'RK'

    brok_lin.compute_transfer_matrices(method, comp_section)

    # When fit is complete, also recompute last elements
    brok_lin.compute_transfer_matrices(method, after_comp_section)

def _select_objective(brok_lin, ref_lin, objective):
    """Assign the fit objective."""
    # Indices of first and last compensating cavities
    first_comp_idx = brok_lin.fault_scenario['comp_idx'][0]
    last_comp_idx = brok_lin.fault_scenario['comp_idx'][-1]

    # Index of synchronous particle at these positions
    first_synch_idx = brok_lin.list_of_elements[first_comp_idx].idx['in']
    last_synch_idx = brok_lin.list_of_elements[last_comp_idx].idx['out']

    dict_objective = {
        'energy': ref_lin.synch.energy['kin_array_mev'][last_synch_idx],
        'phase': ref_lin.synch.phi['abs_array'][last_synch_idx],
        'transfer_matrix': ref_lin.transf_mat['cumul'][last_synch_idx, :, :],
        }
    return dict_objective[objective], first_comp_idx, last_comp_idx


def _select_compensating_cavities(brok_lin, strategy, manual_list=None):
    """
    Return a list of the indexes of compensating cavities.

    Parameters
    ----------
    brok_lin: Accelerator object
        Accelerator with at least one faulty cavity to compensate for.
    strategy: string
        Compensation strategy. Must be 'neighbors', 'all', or 'manual'.
    manual_list: list
        List of compensating cav indices, used when strategy is 'manual'.

    Return
    ------
    comp_idx: list
        List of the indices of the compensating cavities.
    comp_cav: list
        List of the FIELD_MAP objects of the compensating cavities.
        brok_lin.list_of_elements[comp_idx] == comp_cav
    """
    all_idx = []
    working_idx = []
    for i in range(brok_lin.n_elements):
        elt = brok_lin.list_of_elements[i]
        if elt.name == 'FIELD_MAP':
            all_idx.append(i)
            if not elt.status['failed']:
                working_idx.append(i)
    neighbors_idx = _neighbors_of_failed_cav(brok_lin, all_idx)

    dict_strategy = {
        'neighbors': neighbors_idx,
        'all': working_idx,
        'manual': manual_list,
        }

    comp_idx = dict_strategy[strategy]
    comp_cav = [brok_lin.list_of_elements[idx] for idx in comp_idx]
    return comp_idx, comp_cav


def _neighbors_of_failed_cav(brok_lin, all_idx):
    """
    Return a list indices of cavities neighboring failed ones.

    This function tries to return two compensating cavities per faulty one.

    Parameters
    ----------
    brok_lin: Accelerator object
        Broken linac.
    all_idx: list
        List of the indices of all the cavities. If a cavity is missing from
        this list, it will not be compensated if it is faulty, and will not be
        selected for compensation if it is working.

    Return
    ------
    neighbors_idx: list
        List of the indices of the neighbor cavities.
    """
    neighbors_idx = []
    for i in range(len(all_idx)):
        idx = all_idx[i]
        cav = brok_lin.list_of_elements[idx]
        if cav.status['failed']:
            # We select the first working cavities neighboring the i-th cav
            # which are not already in neighbors
            for delta_j in [-1, +1]:
                j = i
                while j in range(0, len(all_idx) + 1):
                    j += delta_j
                    tmp_cav = brok_lin.list_of_elements[all_idx[j]]

                    if tmp_cav.status['failed'] or all_idx[j] in neighbors_idx:
                        j += delta_j
                    else:
                        neighbors_idx.append(all_idx[j])
                        break
    neighbors_idx.sort()
    return neighbors_idx
