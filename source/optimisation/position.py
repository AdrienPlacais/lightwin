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


# !!! fault_idx is still in comp_idx!! FIXME
def compensation_zone(fix: Accelerator, wtf: dict, fault_idx: list[int],
                      comp_idx: list[int]) -> tuple[list[_Element],
                                                    list[_Element]]:
    """Tell what is the zone to recompute."""
    position = wtf['position']
    # FIXME
    if isinstance(position, str):
        position = [position]

    # We need ELEMENT indexes, not cavity.
    fault_idx = _to_elt_idx(fix, fault_idx)
    comp_idx = _to_elt_idx(fix, comp_idx)

    objectives_positions_idx = []
    for pos in position:
        objectives_positions_idx.append(_zone(pos, fix, fault_idx, comp_idx))

    # Take compensation zone that encompasses all individual comp zones
    idx_in = min(fault_idx + comp_idx) - 1
    idx_in = 0
    logging.warning(f"Manually changed {idx_in =}")
    idx_out = max(objectives_positions_idx) + 1

    elts = fix.elts[idx_in:idx_out]
    objectives_positions = [fix.elts[i] for i in objectives_positions_idx]

    return elts, objectives_positions


def _zone(pos: str, *args) -> int | None:
    """Give compensation zone, and position where objectives are checked."""
    if pos not in D_POS:
        logging.error(f"Position {pos} not recognized.""")
        return None
    return D_POS[pos](*args)


def _end_mod(lin: Accelerator, fault_idx: list[int],
             comp_idx: list[int]) -> int:
    """Evaluate obj at the end of the last lattice w/ an altered cavity."""
    idx_last = max(fault_idx + comp_idx)
    idx_lattice_last = lin.elts[idx_last].get('lattice')
    lattices = [lattice for section in lin.get('l_sections', to_numpy=False)
                for lattice in section]
    return lattices[idx_lattice_last][-1].get('elt_idx', to_numpy=False)


def _one_mod_after(lin: Accelerator, fault_idx: list[int],
                   comp_idx: list[int]) -> int:
    """Evaluate objective one lattice after the last comp or failed cav."""
    idx_last = max(fault_idx + comp_idx)
    idx_lattice_last = lin.elts[idx_last].get('lattice') + 1
    l_lattices = [lattice for section in lin.get('l_sections', to_numpy=False)
                  for lattice in section]
    if idx_lattice_last > len(l_lattices):
        logging.warning("You asked for a lattice after the end of the linac. "
                        + "Revert back to previous lattice, i.e. end of "
                        + "linac.")
        idx_lattice_last -= 1
    return l_lattices[idx_lattice_last][-1].get('elt_idx', to_numpy=False)


def _end_linac(lin: Accelerator, fault_idx: list[int],
               comp_idx: list[int]) -> int:
    """Evaluate objective at the end of the linac."""
    return lin.elts[-1].get('elt_idx')


def _to_elt_idx(lin: Accelerator, indexes: list[int]) -> list[int]:
    """Convert list of k-th cavity to list of i-th elements."""
    indexes = [lin.l_cav[i].get('elt_idx', to_numpy=False) for i in indexes]
    return indexes


D_POS = {
    'end_mod': _end_mod,
    '1_mod_after': _end_mod,
    'end_linac': _end_linac,
}
