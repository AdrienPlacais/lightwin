#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 12:41:26 2023.

@author: placais

Here we define the function related to the 'strategy' key of the what_to_fit
dictionary. In particular, it answers the question:
    Given this set of faults, which compensating cavities will be used?

In this module, the indexes are CAVITY indexes, not element.
"""
import logging
import itertools
import math
import numpy as np

from util.helper import flatten
from core.accelerator import Accelerator
from core.list_of_elements import filter_cav

FACT = math.factorial


def sort_and_gather_faults(fix: Accelerator, wtf: dict,
                           fault_idx: list[int] | list[list[int]],
                           comp_idx: list[list[int]] | None = None,
                           ) -> tuple[list[int] | list[list[int]],
                                      list[int] | list[list[int]]]:
    """
    Link faulty cavities with their compensating cavities.

    If two faults need the same compensating cavities, they are gathered, their
    compensating cavities are put in common and they will be fixed together.
    """
    for my_list in [fault_idx, comp_idx]:
        assert _only_field_maps(fix, my_list, idx=wtf['idx'])

        if wtf['idx'] == 'element':
            my_list = _to_cavity_idx(fix, my_list)

    if wtf['strategy'] == 'manual':
        return fault_idx, comp_idx

    gathered_fault_idx, gathered_comp_idx = _gather(fix, fault_idx, wtf)
    return gathered_fault_idx, gathered_comp_idx


def _gather(fix: Accelerator, fault_idx: list[int], wtf: dict
            ) -> tuple[list[list[int]], list[list[int]]]:
    """Gather faults to be fixed together and associated compensating cav."""
    if wtf['strategy'] not in SORT_AND_GATHERERS:
        logging.error('TMP: strategy not reimplemented.')

    fun_sort = SORT_AND_GATHERERS[wtf['strategy']]
    r_comb = 2

    flag_gathered = False
    gathered_faults = [fault_idx]
    while not flag_gathered:
        # List of list of corresp. compensating cavities
        gathered_comp = [fun_sort(fix, faults, wtf)
                         for faults in gathered_faults]

        # Set a counter to exit the 'for' loop when all faults are gathered
        i = 0
        n_combinations = len(gathered_comp)
        if n_combinations <= 1:
            flag_gathered = True
            break
        i_max = int(FACT(n_combinations) / (
            FACT(r_comb) * FACT(n_combinations - r_comb)))

        # Now we look every list of required compensating cavities, and
        # look for faults that require the same compensating cavities
        for ((idx1, l_comp1), (idx2, l_comp2)) \
                in itertools.combinations(enumerate(gathered_comp), r_comb):
            i += 1
            common = list(set(l_comp1) & set(l_comp2))
            # If at least one cavity on common, gather the two
            # corresponding fault and restart the whole process
            if len(common) > 0:
                gathered_faults[idx1].extend(gathered_faults.pop(idx2))
                gathered_comp[idx1].extend(gathered_comp.pop(idx2))
                break

            # If we reached this point, it means that there is no list of
            # faults that share compensating cavities.
            if i == i_max:
                flag_gathered = True

    gathered_comp = [list(filter(lambda idx: idx not in fault_idx,
                                 sublist))
                     for sublist in gathered_comp]
    return gathered_faults, gathered_comp


def _k_neighboring_cavities(lin: Accelerator, fault_idx: list[int],
                            wtf: dict) -> list[int]:
    """Select the cavities neighboring the failed one(s)."""
    # "altered" means compensating or failed cavity
    n_altered_cav = len(fault_idx) * (wtf['k'] + 1)

    # List of distances between the failed cavities and the other ones
    distances = []
    for idx in fault_idx:
        distance = np.array([idx - lin.l_cav.index(cav)
                             for cav in lin.l_cav], dtype=np.float64)
        distances.append(np.abs(distance))
    distances = np.array(distances)
    # Distance between every cavity and it's closest fault
    distance = np.min(distances, axis=0)

    # To favorise cavities near the start of the linac when there is an
    # equality in distance
    sort_bis = np.linspace(1, len(lin.l_cav), len(lin.l_cav))
    # To favorise the cavities near the end of the linac, just invert this:
    # sort_bis = -sort_bis
    # TODO: add a flag in wtf to select this

    idx_altered = np.lexsort((sort_bis, distance))[:n_altered_cav]
    idx_altered.sort()
    idx_altered = list(idx_altered)
    return idx_altered


def _l_neighboring_lattices(lin: Accelerator, fault_idx: list[int],
                            wtf: dict) -> list[int]:
    """Select full lattices neighboring the failed cavities."""
    # For this routine, we use list of faulty cavities instead of list of idx
    fault_cav = [lin.l_cav[idx] for idx in fault_idx]

    cavities_by_lattice = [filter_cav(lattice)
                           for lattice in lin.elts.by_lattice]
    lattices_with_a_fault = [lattice
                             for cav in fault_cav
                             for lattice in cavities_by_lattice
                             if cav in lattice]
    faulty_latt_idx = [cavities_by_lattice.index(lattice)
                       for lattice in lattices_with_a_fault]

    half_n_latt = int(len(fault_idx) * wtf['l'] / 2)
    lattices_with_a_compensating_cavity = \
        cavities_by_lattice[faulty_latt_idx[0] - half_n_latt:
                            faulty_latt_idx[-1] + half_n_latt + 1]

    # List of cavities in the compensating lattices
    idx_compensating = [lin.l_cav.index(cav)
                        for lattice in lattices_with_a_compensating_cavity
                        for cav in lattice]
    return idx_compensating


def _all_cavities(lin: Accelerator, fault_idx: list[int], wtf: dict
                  ) -> list[int]:
    """Select all the cavities of the linac."""
    cavities = lin.l_cav
    idx_altered = [idx for idx in range(len(cavities))]
    return idx_altered


def _all_cavities_after_first_failure(lin: Accelerator, fault_idx: list[int],
                                      wtf: dict) -> list[int]:
    """Select all the cavities after the first failed cavity."""
    altered_cavities = lin.l_cav[min(fault_idx):]
    idx_altered = [lin.l_cav.index(cav) for cav in altered_cavities]
    return idx_altered


def _only_field_maps(lin: Accelerator,
                     indexes: list[int] | list[list[int]] | None,
                     idx: str) -> bool:
    """Check that all the required elements are cavities."""
    if indexes is None:
        return True

    elts = lin.elts
    if idx == 'cavity':
        elts = lin.l_cav

    natures = set([elts[i].get('nature') for i in flatten(indexes)])
    if natures != {'FIELD_MAP'}:
        logging.error("Some elements are not cavities.")
        return False
    return True


def _to_cavity_idx(lin: Accelerator,
                   indexes: list[int] | list[list[int]] | None
                   ) -> list[int] | list[list[int]] | None:
    """
    Convert i-th element to k_th cavity.

    Works with list of indexes (ungathered) and list of list of indexes
    (gathered, which is when method = 'manual'.
    """
    if indexes is None:
        return None

    set_types = {type(idx) for idx in indexes}
    list_in = lin.elts
    list_out = lin.l_cav

    if set_types == {int}:
        elts = [list_in[i] for i in indexes]
        indexes = [list_out.index(elt) for elt in elts]
        return indexes

    if set_types == {list}:
        grouped_elts = [[list_in[i] for i in idx] for idx in indexes]
        grouped_indexes = [[list_out.index(elt) for elt in elts]
                           for elts in grouped_elts]
        return grouped_indexes

    logging.error(f"{indexes} data type was not recognized.")
    return None


SORT_AND_GATHERERS = {
    'k out of n': _k_neighboring_cavities,
    'l neighboring lattices': _l_neighboring_lattices,
    'global': _all_cavities,
    'global downstream': _all_cavities_after_first_failure,
}
