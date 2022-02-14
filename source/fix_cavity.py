#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 12:51:15 2022

@author: placais
"""


def apply_faults(broken_linac, idx_fail_cav):
    """Break cavities at indices idx_fail_cav."""
    broken_linac.fault_scenario['idx_faults'] = idx_fail_cav

    for idx in idx_fail_cav:
        cavity = broken_linac.list_of_elements[idx]
        assert cavity.name == 'FIELD_MAP', 'Error, the element at ' + \
            'position ' + str(idx) + ' is not a FIELD_MAP.'
        cavity.fail()
        broken_linac.fault_scenario['cav_faults'].append(cavity)


def compensate_faults(broken_linac, ref_linac, objective_str, strategy,
                      manual_list=None):
    """Compensate faults, according to strategy and objective."""
    # Select which cavities will be used to compensate the fault
    broken_linac.fault_scenario['strategy'] = strategy
    broken_linac.fault_scenario['cav_compensating'] = \
        broken_linac._select_compensating_cavities(strategy, manual_list)

    for elt in broken_linac.fault_scenario['cav_compensating']:
        elt.status['compensate'] = True

    broken_linac.fault_scenario['objective_str'] = objective_str
    broken_linac.fault_scenario['objective'] = \
        _select_objective(ref_linac, objective_str)


def _select_compensating_cavities(broken_linac, strategy, manual_list):
    """Return a list of the indexes of compensating cavities."""
    all_cav = list(filter(lambda elt: elt.name == 'FIELD_MAP',
                          broken_linac.list_of_elements))
    working_cav = list(filter(lambda elt: elt.status['failed'] is False,
                              all_cav))
    neighbors_cav = _neighbors_of_failed_cav(all_cav)

    dict_strategy = {
        'neighbors': neighbors_cav,
        'all': working_cav,
        'manual': manual_list,
        }
    return dict_strategy[strategy]


def _select_objective(ref_linac, objective):
    """Assign what will be fitted."""
    dict_objective = {
        'energy': ref_linac.synch.energy['kin_array_mev'][-1],
        'phase': ref_linac.synch.phi['abs_array'][-1],
        'transfer_matrix': ref_linac.transf_mat['cumul'][-1, :, :],
        }
    return dict_objective[objective]


def _neighbors_of_failed_cav(list_of_cav):
    """Return a list of the cavities neighboring failed ones."""
    neighbors = []
    for i in range(len(list_of_cav)):
        if list_of_cav[i].status['failed']:
            # We select the first working cavities neighboring the i-th cav
            # which are not already in neighbors
            for delta_j in [-1, +1]:
                j = i
                while j in range(0, len(list_of_cav) + 1):
                    j += delta_j

                    if list_of_cav[j].status['failed'] or j in neighbors:
                        j += delta_j
                    else:
                        neighbors.append(j)
                        break
    neighbors.sort()
    return [list_of_cav[idx] for idx in neighbors]
