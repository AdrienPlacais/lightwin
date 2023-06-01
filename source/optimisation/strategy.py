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
from core.elements import FieldMap

FACT = math.factorial


def sort_and_gather_faults(
        fix: Accelerator, wtf: dict,
        l_fault_idx: list[int, ...] | list[list[int, ...], ...],
        l_comp_idx: list[list[int, ...], ...] | None = None,
) -> tuple[list[int, ...] | list[list[int, ...], ...],
           list[int, ...] | list[list[int, ...], ...]]:
    """
    Link faulty cavities with their compensating cavities.

    If two faults need the same compensating cavities, they are gathered, their
    compensating cavities are put in common and they will be fixed together.
    """
    # Check nature, convert to CAVITY index if necessary
    for my_list in [l_fault_idx, l_comp_idx]:
        assert _only_field_maps(fix, my_list, idx=wtf['idx'])

        if wtf['idx'] == 'element':
            my_list = _to_cavity_idx(fix, my_list)

    # If manual, we are done
    if wtf['strategy'] == 'manual':
        return l_fault_idx, l_comp_idx

    # If not, gather and sort with proper method
    ll_fault_idx, ll_comp_idx = _gather(fix, l_fault_idx, wtf)
    return ll_fault_idx, ll_comp_idx


def _gather(fix: Accelerator, l_fault_idx: list[int, ...], wtf: dict
            ) -> tuple[list[list[int, ...], ...], list[list[int, ...], ...]]:
    """Gather faults to be fixed together and associated compensating cav."""
    if wtf['strategy'] not in D_SORT_AND_GATHER:
        logging.error('TMP: strategy not reimplemented.')

    fun_sort = D_SORT_AND_GATHER[wtf['strategy']]
    r_comb = 2

    flag_gathered = False
    ll_fault = [l_fault_idx]
    while not flag_gathered:
        # List of list of corresp. compensating cavities
        ll_comp = [fun_sort(fix, l_fault, wtf) for l_fault in ll_fault]

        # Set a counter to exit the 'for' loop when all faults are gathered
        i = 0
        n_comb = len(ll_comp)
        if n_comb <= 1:
            flag_gathered = True
            break
        i_max = int(FACT(n_comb) / (FACT(r_comb) * FACT(n_comb - r_comb)))

        # Now we look every list of required compensating cavities, and
        # look for faults that require the same compensating cavities
        for ((idx1, l_comp1), (idx2, l_comp2)) \
                in itertools.combinations(enumerate(ll_comp), r_comb):
            i += 1
            common = list(set(l_comp1) & set(l_comp2))
            # If at least one cavity on common, gather the two
            # corresponding fault and restart the whole process
            if len(common) > 0:
                ll_fault[idx1].extend(ll_fault.pop(idx2))
                ll_comp[idx1].extend(ll_comp.pop(idx2))
                break

            # If we reached this point, it means that there is no list of
            # faults that share compensating cavities.
            if i == i_max:
                flag_gathered = True

    return ll_fault, ll_comp


def _k_neighboring_cavities(lin: Accelerator, l_fault_idx: list[int, ...],
                            wtf: dict) -> list[int, ...]:
    """Select the cavities neighboring the failed one(s)."""
    n_comp_cav = len(l_fault_idx) * (wtf['k'] + 1)

    # List of distances between the failed cavities and the other ones
    distances = []
    for idx_f in l_fault_idx:
        distance = np.array([idx_f - lin.l_cav.index(cav)
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

    idx_compensating = np.lexsort((sort_bis, distance))[:n_comp_cav]
    return list(idx_compensating.sort())


def _l_neighboring_lattices(lin: Accelerator, l_faulty_idx: list[int, ...],
                            wtf: dict) -> list[int, ...]:
    """Select full lattices neighboring the failed cavities."""
    # For this routine, we use list of faulty cavities instead of list of idx
    l_faulty_cav = [lin.l_cav[idx] for idx in l_faulty_idx]

    # List of cavities, sorted by lattice
    ll_all_latt = [filter_cav(lattice)
                   for section in lin.get('l_sections', to_numpy=False)
                   for lattice in section]
    # List of lattices with a fault
    ll_faulty_latt = [lattice
                      for faulty_cav in l_faulty_cav
                      for lattice in ll_all_latt
                      if faulty_cav in lattice]
    faulty_latt_idx = [ll_all_latt.index(lattice)
                       for lattice in ll_faulty_latt]

    half_n_latt = int(len(l_faulty_idx) * wtf['l'] / 2)
    # List of compensating lattices
    l_comp_latt = ll_all_latt[faulty_latt_idx[0] - half_n_latt:
                              faulty_latt_idx[-1] + half_n_latt + 1]

    # List of cavities in the compensating lattices
    idx_compensating = [lin.l_cav.index(cav)
                        for lattice in l_comp_latt
                        for cav in lattice]
    return idx_compensating


def _only_field_maps(lin: Accelerator,
                     ll_idx: list[int, ...] | list[list[int, ...], ...] | None,
                     idx: str) -> bool:
    """Check that all the required elements are cavities."""
    if ll_idx is None:
        return True

    l_elts = lin.elts
    if idx == 'cavity':
        l_elts = lin.l_cav
    # natures = {x.get('nature') for x in flatten(ll_cav)}
    # if natures != {'FIELD_MAP'}:
    types = {type(l_elts[i]) for i in flatten(ll_idx)}
    if types != {FieldMap}:
        logging.error("Some elements are not cavities.")
        return False
    return True


def _to_cavity_idx(lin: Accelerator,
                   l_idx: list[int, ...] | list[list[int, ...]] | None
                  ) -> list[int, ...] | list[list[int, ...]] | None:
    """
    Convert i-th element to k_th cavity.

    Works with list of indexes (ungathered) and list of list of indexes
    (gathered, which is when method = 'manual'.
    """
    if l_idx is None:
        return None

    set_types = {type(idx) for idx in l_idx}
    list_in = lin.elts
    list_out = lin.l_cav

    if set_types == {int}:
        l_elts = [list_in[i] for i in l_idx]
        l_idx = [list_out.index(elt) for elt in l_elts]
        return l_idx

    if set_types == {list}:
        ll_elts = [[list_in[i] for i in idx] for idx in l_idx]
        ll_idx = [[list_out.index(elt) for elt in l_elts]
                  for l_elts in ll_elts]
        return ll_idx

    logging.error(f"{l_idx} data type was not recognized.")
    return None


D_SORT_AND_GATHER = {
    'k out of n': _k_neighboring_cavities,
    'l neighboring lattices': _l_neighboring_lattices,
}
