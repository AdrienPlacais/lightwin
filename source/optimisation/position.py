#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:21:37 2023

@author: placais

Here we define the function related to the 'position' key of the what_to_fit
dictionary. In particular, it answers the question:
    Given this set of faults and compensating cavities, what is the portion of
    the linac that I need to recompute again and again during the fit?

In this module, the indexes are ELEMENT indexes, not cavity.
In contrary to the optimisation.strategy module, here we do one fault at a time
(a fault can encompass several faulty cavities that are to be fixed together).

TODO (both), end_section, elt_name
TODO remove the position list checking once it is implemented
"""
import logging

from core.accelerator import Accelerator
from core.elements import _Element


def compensation_zone(fix: Accelerator, wtf: dict, l_fault_idx: list[int],
                      l_comp_idx: list[int]
                     ) -> tuple[list[_Element], list[int]]:
    """Tell what is the zone to recompute."""
    position = wtf['position']
    # FIXME
    if isinstance(position, str):
        position = [position]

    # We need ELEMENT indexes, not cavity.
    for l_idx in [l_fault_idx, l_comp_idx]:
        l_idx = _to_elt_idx(fix, l_idx)

    l_check = []
    for pos in position:
        l_check.append(_zone(pos))

    # Take compensation zone that encompasses all individual comp zones
    idx_in = min(l_fault_idx + l_comp_idx) - 1
    idx_out = max(l_check) + 1
    l_elts = fix.elts[idx_in, idx_out]

    return l_elts, l_check


def _zone(pos: str) -> int | None:
    """Give compensation zone, and position where objectives are checked."""
    if pos not in D_POS:
        logging.error(f"Position {pos} not recognized.""")
        return None
    return D_POS[pos]


def _end_mod(lin: Accelerator, l_fidx: list[int], l_cidx: list[int]) -> int:
    """Evaluate objective at the end of the last lattice w/ an altered cavity.
    """
    idx_last = max(l_fidx + l_cidx)
    idx_lattice_last = lin.elts[idx_last].get('lattice')
    l_lattices = [lattice for section in lin.get('l_sections', to_numpy=False)
                  for lattice in section]
    return l_lattices[idx_lattice_last][-1].get('elt_idx')


def _one_mod_after(lin: Accelerator, l_fidx: list[int],
                   l_cidx: list[int]) -> int:
    """Evaluate objective one lattice after the last comp or failed cav."""
    idx_last = max(l_fidx + l_cidx)
    idx_lattice_last = lin.elts[idx_last].get('lattice') + 1
    l_lattices = [lattice for section in lin.get('l_sections', to_numpy=False)
                  for lattice in section]
    if idx_lattice_last > len(l_lattices):
        logging.warning("You asked for a lattice after the end of the linac. "
                        + "Revert back to previous lattice, i.e. end of "
                        + "linac.")
        idx_lattice_last -= 1
    return l_lattices[idx_lattice_last][-1].get('elt_idx')


def _end_linac(lin: Accelerator, l_fidx: list[int], l_cidx: list[int]) -> int:
    """Evaluate objective at the end of the linac."""
    return lin.elts[-1].get('elt_idx')


def _to_elt_idx(lin: Accelerator, l_idx: list[int]) -> list[int]:
    """Convert list of k-th cavity to list of i-th elements. """
    l_idx = [lin.l_cav[i].get('elt_idx', to_numpy=False) for i in l_idx]
    return l_idx


D_POS = {
    'end_mod': _end_mod,
    '1_mod_after': _end_mod,
    'end_linac': _end_linac,
}
