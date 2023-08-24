#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 18 12:21:37 2023.

@author: placais

Here we define the function related to the 'position' key of the
``what_to_fit`` dictionary. In particular, it answers the question:
given this set of faults and compensating cavities, what is the portion of
the linac that I need to recompute again and again during the fit?

.. important::
    In this module, the indexes are ELEMENT indexes, not cavity.

.. important::
    In contrary to the :mod:`failures.strategy` module, here we do one
    fault at a time (a fault can encompass several faulty cavities that are to
    be fixed together).

.. todo::
    end_section, elt_name

"""
import logging

from core.accelerator import Accelerator
from core.elements import _Element


def compensation_zone(fix: Accelerator, wtf: dict, fault_idx: list[int],
                      comp_idx: list[int], need_full_lattices: bool = False
                      ) -> tuple[list[_Element], list[_Element]]:
    """
    Tell what is the zone to recompute.

    We use in this routine element indexes, not cavity indexes.

    Parameters
    ----------
    fix : Accelerator
        `Accelerator` beeing fixed.
    wtf : dict
        Holds information on what to fit.
    fault_idx : list[int]
        Cavity index of the faults, directly converted to element index in the
        routine.
    comp_idx : list[int]
        Cavity index of the compensating cavities, directly converted to
        element index in the routine.
    need_full_lattices : bool, optional
        If you want the compensation zone to encompass full lattices only. It
        is a little bit slower as more ``_Element`` are calculated. Plus,
        it has no impact even with :class:`TraceWin` solver. Keeping it in case
        it has an impact that I did not see.

    Returns
    -------
    elts : list[_Element]
        ``_Element`` objects of the compensation zone.
    objectives_positions : list[int]
        Position in ``elts`` of where objectives should be matched.

    """
    position = wtf['position']

    fault_idx = _to_elt_idx(fix, fault_idx)
    comp_idx = _to_elt_idx(fix, comp_idx)

    objectives_positions_idx = [_zone(pos, fix, fault_idx, comp_idx)
                                for pos in position]

    idx_start_compensation_zone = min(fault_idx + comp_idx)

    if need_full_lattices:
        idx_start_compensation_zone = \
            _reduce_idx_start_to_include_full_lattice(
                idx_start_compensation_zone,
                fix)

    idx_end_compensation_zone = max(objectives_positions_idx)

    elts = fix.elts[idx_start_compensation_zone:idx_end_compensation_zone + 1]
    objectives_positions = [fix.elts[i] for i in objectives_positions_idx]

    return elts, objectives_positions


def _zone(pos: str, *args) -> int | None:
    """Give compensation zone, and position where objectives are checked."""
    if pos not in POSITION_TO_INDEX:
        logging.error(f"Position {pos} not recognized.""")
        return None
    return POSITION_TO_INDEX[pos](*args)


def _end_last_altered_lattice(lin: Accelerator, fault_idx: list[int],
                              comp_idx: list[int]) -> int:
    """Evaluate obj at the end of the last lattice w/ an altered cavity."""
    idx_last = max(fault_idx + comp_idx)
    idx_lattice_last = lin.elts[idx_last].get('lattice')
    idx_eval = lin.elts.by_lattice[idx_lattice_last][-1].get('elt_idx',
                                                             to_numpy=False)
    return idx_eval


def _one_lattice_after_last_altered_lattice(
        lin: Accelerator, fault_idx: list[int], comp_idx: list[int]) -> int:
    """Evaluate objective one lattice after the last comp or failed cav."""
    idx_last = max(fault_idx + comp_idx)
    idx_lattice_last = lin.elts[idx_last].get('lattice') + 1
    if idx_lattice_last > len(lin.elts.by_lattice):
        logging.warning("You asked for a lattice after the end of the linac. "
                        + "Revert back to previous lattice, i.e. end of "
                        + "linac.")
        idx_lattice_last -= 1
    idx_eval = lin.elts.by_lattice[idx_lattice_last][-1].get('elt_idx',
                                                             to_numpy=False)
    return idx_eval


def _end_last_failed_lattice(lin: Accelerator, fault_idx: list[int],
                             comp_idx: list[int]) -> int:
    """Evaluate obj at the end of the last lattice w/ a failed cavity."""
    idx_last = max(fault_idx)
    idx_lattice_last = lin.elts[idx_last].get('lattice')
    idx_eval = lin.elts.by_lattice[idx_lattice_last][-1].get('elt_idx',
                                                             to_numpy=False)
    return idx_eval


def _end_linac(lin: Accelerator, fault_idx: list[int],
               comp_idx: list[int]) -> int:
    """Evaluate objective at the end of the linac."""
    return lin.elts[-1].get('elt_idx')


def _reduce_idx_start_to_include_full_lattice(idx: int, lin: Accelerator
                                              ) -> int:
    """Force compensation zone to start at the 1st element of lattice."""
    logging.warning("Changed compensation zone to include full lattices only.")
    elt = lin.elts[idx]
    lattice_idx = elt.get('lattice', to_numpy=False)
    elt = lin.elts.by_lattice[lattice_idx][0]
    idx = elt.get('elt_idx', to_numpy=False)
    return idx


def _to_elt_idx(lin: Accelerator, indexes: list[int]) -> list[int]:
    """Convert list of k-th cavity to list of i-th elements."""
    indexes = [lin.l_cav[i].get('elt_idx', to_numpy=False) for i in indexes]
    return indexes


POSITION_TO_INDEX = {
    'end of last altered lattice': _end_last_altered_lattice,
    'one lattice after last altered lattice':
        _one_lattice_after_last_altered_lattice,
    'end of last failed lattice': _end_last_failed_lattice,
    'end of linac': _end_linac,
}
