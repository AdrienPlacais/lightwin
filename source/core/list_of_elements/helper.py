#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define some helper functions to filter list of elements.

.. todo::
    Filtering consistency.

"""
import logging
from functools import partial
from typing import Any, Sequence, Type, TypeGuard, TypeVar

import numpy as np

from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap

# from core.list_of_elements.list_of_elements import ListOfElements

ListOfElements = TypeVar("ListOfElements")


def is_list_of(elts: Sequence, type_to_check: Type) -> TypeGuard[Type]:
    """Check that all items of ``elts`` are of type ``type_to_check``."""
    if not hasattr(elts, "__iter__"):
        return False
    return all([isinstance(elt, type_to_check) for elt in elts])


def is_list_of_elements(elts: Sequence) -> TypeGuard[list[Element]]:
    """Check that all elements of input are :class:`.Element`."""
    # if isinstance(elts, ListOfElements):
    #     return True
    if hasattr(elts, "input_particle"):
        return True
    return is_list_of(elts, Element)


def is_list_of_list_of_elements(
    elts: Sequence,
) -> TypeGuard[list[list[Element]]]:
    """Check that input is a nested list of :class:`.Element`."""
    return all([is_list_of_elements(sub_elts) for sub_elts in elts])


def filter_out(
    elts: ListOfElements | Sequence[Element] | Sequence[Sequence[Element]],
    to_exclude: tuple[type],
) -> Any:
    """Filter out types while keeping the input list structure.

    .. note::
        Function not used anymore. Keeping it just in case.

    """
    if is_list_of_elements(elts):
        out = list(filter(lambda elt: not isinstance(elt, to_exclude), elts))

    elif is_list_of_list_of_elements(elts):
        out = [filter_out(sub_elts, to_exclude) for sub_elts in elts]

    else:
        raise TypeError("Wrong type for data filtering.")

    assert isinstance(out, type(elts))
    return out


def filter_elts(
    elts: ListOfElements | Sequence[Element], type_to_check: Type
) -> list[Type]:
    """Filter elements according to their type.

    .. note::
        Used only for :func:`filter_cav`, may be simpler?

    """
    return list(filter(lambda elt: isinstance(elt, type_to_check), elts))


filter_cav = partial(filter_elts, type_to_check=FieldMap)


def elt_at_this_s_idx(
    elts: ListOfElements | Sequence[Element],
    s_idx: int,
    show_info: bool = False,
) -> Element | None:
    """Give the element where the given index is.

    Parameters
    ----------
    elts : ListOfElements | list[Element]
        List of elements in which to look for.
    s_idx : int
        Index to look for.
    show_info : bool
        If the element that we found should be outputed.

    Returns
    -------
    elt : Element | None
        Element where the mesh index ``s_idx`` is in ``elts``.

    """
    for elt in elts:
        if s_idx in range(elt.idx["s_in"], elt.idx["s_out"]):
            if show_info:
                logging.info(
                    f"Mesh index {s_idx} is in {elt.get('elt_info')}.\n"
                    f"Indexes of this elt: {elt.get('idx')}."
                )
            return elt

    logging.warning(f"Mesh index {s_idx} not found.")
    return None


def equivalent_elt_idx(
    elts: ListOfElements | list[Element], elt: Element | str
) -> int:
    """Return the index of element from ``elts`` corresponding to ``elt``.

    .. important::
        This routine uses the name of the element and not its adress. So
        it will not complain if the :class:`.Element` object that you asked for
        is not in this list of elements.
        In the contrary, it was meant to find equivalent cavities between
        different lists of elements.

    See also
    --------
    :func:`equivalent_elt`
    :meth:`.Accelerator.equivalent_elt`

    Parameters
    ----------
    elts : ListOfElements | list[Element]
        List of elements where you want the position.
    elt : Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an element. If it is an :class:`.Element`, we take its name
        in the routine. Magic keywords ``'first'``, ``'last'`` are also
        accepted.

    Returns
    -------
    int
        Index of equivalent element.

    """
    if not isinstance(elt, str):
        elt = elt.name

    magic_keywords = {"first": 0, "last": -1}
    names = [x.name for x in elts]

    if elt in names:
        return names.index(elt)

    if elt in magic_keywords:
        return magic_keywords[elt]

    logging.error(f"Element {elt} not found in this list of elements.")
    logging.debug(f"List of elements is:\n{elts}")
    raise IOError(f"Element {elt} not found in this list of elements.")


def equivalent_elt(
    elts: ListOfElements | list[Element], elt: Element | str
) -> Element:
    """Return the element from ``elts`` corresponding to ``elt``.

    .. important::
        This routine uses the name of the element and not its adress. So
        it will not complain if the :class:`.Element` object that you asked for
        is not in this list of elements.
        In the contrary, it was meant to find equivalent cavities between
        different lists of elements.

    See also
    --------
    :func:`equivalent_elt_idx`
    :meth:`.Accelerator.equivalent_elt`

    Parameters
    ----------
    elts : ListOfElements | list[Element]
        List of elements where you want the position.
    elt : Element | str
        Element of which you want the position. If you give a str, it should be
        the name of an element. If it is an :class:`.Element`, we take its name
        in the routine. Magic keywords ``'first'``, ``'last'`` are also
        accepted.

    Returns
    -------
    out_elt : Element
        Equivalent element.

    """
    out_elt_idx = equivalent_elt_idx(elts, elt)
    out_elt = elts[out_elt_idx]
    return out_elt


def indiv_to_cumul_transf_mat(
    tm_cumul_in: np.ndarray, r_zz_elt: list[np.ndarray], n_steps: int
) -> np.ndarray:
    """Compute cumulated transfer matrix.

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
        cumulated_transfer_matrices[i] = (
            r_zz_elt[i - 1] @ cumulated_transfer_matrices[i - 1]
        )
    return cumulated_transfer_matrices


def group_elements_by_section(elts: Sequence[Element]) -> list[list[Element]]:
    """Group elements by section."""
    n_sections = elts[-1].idx["section"] + 1
    by_section = [
        list(filter(lambda elt: elt.idx["section"] == current_section, elts))
        for current_section in range(n_sections)
    ]
    return by_section


def group_elements_by_section_and_lattice(
    by_section: Sequence[Sequence[Element]],
) -> list[list[list[Element]]]:
    """Regroup Elements by Section and then by Lattice."""
    by_section_and_lattice = [
        group_elements_by_lattice(section) for section in by_section
    ]
    return by_section_and_lattice


def group_elements_by_lattice(elts: Sequence[Element]) -> list[list[Element]]:
    """Regroup the Element belonging to the same Lattice."""
    idx_first_lattice = elts[0].idx["lattice"]
    idx_last_lattice = elts[-1].idx["lattice"]
    n_lattices = idx_last_lattice + 1
    by_lattice = [
        list(
            filter(
                lambda elt: (
                    elt.idx["lattice"] >= 0
                    and elt.idx["lattice"] == current_lattice
                ),
                elts,
            )
        )
        for current_lattice in range(idx_first_lattice, n_lattices)
    ]
    return by_lattice
