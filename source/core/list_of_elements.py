#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 15:11:55 2022.

@author: placais

In this module we define the ListOfElements objects. These objects are created
in two contexts:
    - Accelerator.elts: holds all the _Elements of the linac.
    - Fault.elts, also called in FaultScenario: it holds only a fraction of the
    linac _Elements. Beam will be propagated a huge number of time during
    optimisation process, so we recompute only the strict necessary.
"""
import logging
from typing import Any
from functools import partial
import numpy as np

from util.helper import recursive_items, recursive_getter
from core.elements import _Element


class ListOfElements(list):
    """Class holding the elements of a fraction or of the whole linac."""

    def __init__(self, l_elts: list[_Element], w_kin: float, phi_abs: float,
                 tm_cumul: np.ndarray | None = None, full_linac: bool = True
                 ) -> None:
        """
        Create the object, encompassing all the linac or only a fraction.

        The first case is when you initialize an Accelerator and compute the
        baseline energy, phase, etc values.
        The second case is when you only recompute a fraction of the linac,
        which is part of the optimisation process.

        Parameters
        ----------
        l_elts : list[_Element]
            List containing the _Element objects.
        w_kin : float
            Kinetic energy at the entrance of the first _Element. Must be in
            MeV.
        phi_abs : float
            Absolute phase at the entrance of the first _Element. Must be in
            rad, and expressed relative to the bunch frequency.
        tm_cumul : np.ndarray, optional
            Cumulated transfer matrix (2, 2) at the entrance of the first
            _Element. The default is None.
        full_linac : bool, optional
            To indicate if this a full linac or only a portion (fit process).
            The default is True.

        """
        if full_linac:
            logging.info("Init ListOfElements ecompassing all linac.")
            if tm_cumul is not None:
                logging.warning(
                    "You do not need to provide a cumulated transfer matrix "
                    + "to initialize this ListOfElements. It starts at the "
                    + "beginning of the linac and the matrix is automatically "
                    + "the eye matrix.")
            tm_cumul = np.eye(2)

        else:
            logging.info(f"Init ListOfElements from {l_elts[0]} to "
                         + f"{l_elts[-1]}.")
            if np.any(np.isnan(tm_cumul)):
                logging.error("Previous transfer matrix was not calculated.")

        super().__init__(l_elts)
        logging.info(f"{w_kin = }, {phi_abs = }")
        self.w_kin_in = w_kin
        self.phi_abs_in = phi_abs

        self.tm_cumul_in = tm_cumul
        self._l_cav = filter_cav(self)

    @property
    def l_cav(self):
        """Easy access to the list of cavities."""
        return self._l_cav

    def has(self, key: str) -> bool:
        """Tell if the required attribute is in this class."""
        return key in recursive_items(vars(self)) or \
            key in recursive_items(vars(self[0]))

    def get(self, *keys: tuple[str], to_numpy: bool = True,
            remove_first: bool = False, **kwargs: dict) -> Any:
        """Shorthand to get attributes."""
        val = {}
        for key in keys:
            val[key] = []

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue

            # Specific case: key is in Element so we concatenate all
            # corresponding values in a single list
            if self[0].has(key):
                for elt in self:
                    data = elt.get(key, to_numpy=False, **kwargs)
                    # In some arrays such as z position arrays, the last pos of
                    # an element is the first of the next
                    if remove_first and elt is not self[0]:
                        data = data[1:]
                    if isinstance(data, list):
                        val[key] += data
                    else:
                        val[key].append(data)
            else:
                val[key] = recursive_getter(key, vars(self), **kwargs)

        # Convert to list, and to numpy array if necessary
        out = [val[key] for key in keys]
        if to_numpy:
            out = [np.array(val) for val in out]

        # Return as tuple or single value
        if len(keys) == 1:
            return out[0]
        # implicit else
        return tuple(out)


def indiv_to_cumul_transf_mat(tm_cumul_in: np.ndarray,
                              r_zz_elt: list[np.ndarray], n_steps: int
                              ) -> np.ndarray:
    """
    Compute cumulated transfer matrix.

    Parameters
    ----------
    tm_cumul_in : np.ndarray
        Cumulated transfer matrix @ first element. Should be eye matrix if we
        are at the first element.
    r_zz_elt : list[np.ndarray]
        List of individual transfer matrix of the elements.
    n_steps : int
        Number of elements or elements slices.

    Returns
    -------
    cumulated_transfer_matrices : np.ndarray
        Cumulated transfer matrices.

    """
    cumulated_transfer_matrices = np.full((n_steps, 2, 2), np.NaN)
    cumulated_transfer_matrices[0] = tm_cumul_in
    for i in range(1, n_steps):
        cumulated_transfer_matrices[i] = \
            r_zz_elt[i - 1] @ cumulated_transfer_matrices[i - 1]
    return cumulated_transfer_matrices


def equiv_elt(elts: ListOfElements | list[_Element, ...],
              elt: _Element | str, to_index: bool = False
              ) -> _Element | int | None:
    """
    Return an element from elts that has the same name as elt.

    Important: this routine uses the name of the element and not its adress. So
    it will not complain if the _Element object that you asked for is not in
    this list of elements.
    In the contrary, it was meant to find equivalent cavities between different
    lists of elements.

    Parameters
    ----------
    elts : ListOfElements | list[_Element, ...]
        List of elements where you want the position.
    elt : _Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an _Element. If it is an _Element, we take its name in the
        routine.
    to_index : bool, optional
        If True, the function returns the index of the _Element instead of the
        _Element itself.

    Returns
    -------
    _Element | int | None
        Equivalent _Element, position in list of elements, or None if not
        found.

    """
    if not isinstance(elt, str):
        elt = elt.get("elt_name")

    names = [x.get("elt_name") for x in elts]
    if elt not in names:
        logging.error(f"Element {elt} not found in this list of elements.")
        logging.debug(f"List of elements is:\n{elts}")
        return None

    idx = names.index(elt)
    if to_index:
        return idx
    return elts[idx]


def elt_at_this_s_idx(elts: ListOfElements | list[_Element, ...],
                      s_idx: int, show_info: bool = False
                      ) -> _Element | None:
    """Give the element where the given index is."""
    for elt in elts:
        if s_idx in range(elt.idx['s_in'], elt.idx['s_out']):
            if show_info:
                logging.info(
                    f"Mesh index {s_idx} is in {elt.get('elt_info')}.\n"
                    + f"Indexes of this elt: {elt.get('idx')}.")
            return elt

    logging.warning(f"Mesh index {s_idx} not found.")
    return None


def filter_elts(elts: ListOfElements | list[_Element, ...], key: str, val: Any
                ) -> list[_Element, ...]:
    """Shortcut for filtering elements according to (key, val)."""
    return list(filter(lambda elt: elt.get(key) == val, elts))


filter_cav = partial(filter_elts, key='nature', val='FIELD_MAP')
