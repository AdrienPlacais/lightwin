#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define some helper functions to filter list of elements.

.. todo::
    Filtering consistency.

"""
import logging
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Sequence, Type, TypeGuard, TypeVar

import numpy as np

from core.elements.element import Element
from core.elements.field_maps.field_map import FieldMap

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


def is_list_of_list_of_field_maps(
    elts: Sequence,
) -> TypeGuard[list[list[FieldMap]]]:
    """Check that input is a nested list of :class:`.Element`."""
    return all([is_list_of(sub_elts, FieldMap) for sub_elts in elts])


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


def group_elements_by_section(
    elts: Sequence[Element], n_to_check: int = 10
) -> Sequence[Sequence[Element]]:
    """Group elements by section."""
    n_sections = (
        _get_first_key_of_idx_dict_higher_than(
            elts,
            index_name="section",
            first_or_last="last",
            higher_than=-1,
            n_to_check=n_to_check,
        )
        + 1
    )
    if n_sections <= 0:
        return [elts]

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
    idx_first_lattice = _get_first_key_of_idx_dict_higher_than(
        elts, index_name="lattice", first_or_last="first", higher_than=-1
    )
    idx_last_lattice = _get_first_key_of_idx_dict_higher_than(
        elts, index_name="lattice", first_or_last="last", higher_than=-1
    )
    n_lattices = idx_last_lattice + 1
    by_lattice = [
        list(
            filter(
                lambda elt: (
                    elt.increment_lattice_idx
                    and elt.idx["lattice"] == current_lattice
                ),
                elts,
            )
        )
        for current_lattice in range(idx_first_lattice, n_lattices)
    ]
    return by_lattice


def _get_first_key_of_idx_dict_higher_than(
    elts: Sequence[Element],
    *,
    index_name: str,
    first_or_last: str,
    higher_than: int = -1,
    n_to_check: int = 10,
) -> int:
    """Take first valid idx in ``n_to_check`` first/last elements of ``elts``.

    Typical usage is getting the number of sections or lattice by taking the
    last element with a section/lattice index higher than -1.

    Parameters
    ----------
    elts : Sequence[Element]
        List of elements to check.
    index_name : str
        Name of the index to get. Must be a key of :attr:`.Element.idx`.
    first_or_last : {'first', 'last'}
        If we want to check the ``n_to_check`` first or last elements.
    higher_than : int, optional
        The index under which the value is invalid. The default is -1, which is
        the initialisation index for all the values of the ``idx`` dictionary.
    n_to_check : int, optional
        Number of elements in which we will look for the index.

    Returns
    -------
    int
        The first valid index that is found.

    """
    assert first_or_last in ("first", "last")
    elts_to_check = elts[: n_to_check + 1]
    if first_or_last == "last":
        elts_to_check = elts[: -n_to_check - 1 : -1]

    index = -1
    for elt in elts_to_check:
        index = elt.idx.get(index_name, None)
        if index is None:
            logging.warning(
                f"Could not find key {index_name} in {elt} idx dictionary "
                f"because it does not exist. {elt.idx = }."
            )
            continue
        if index > higher_than:
            return index

    logging.warning(
        f"There is no element with an attribute idx['{index_name}'] higher "
        f"than {higher_than} in the {n_to_check} {first_or_last} elements of "
        "provided list of elements."
    )
    return -1


def first[
    T
](
    iterable: Iterable[T],
    default: T | None = None,
    condition: Callable[[T], bool] = lambda _: True,
) -> T:
    """Return the first item in ``iterable`` satisfying ``condition``.

    If the condition is not given, returns the first item of
    the iterable.

    If the ``default`` argument is given and the iterable is empty,
    or if it has no items matching the condition, the `default` argument
    is returned if it matches the condition.

    The ``default`` argument being None is the same as it not being given.

    Raises ``StopIteration`` if no item satisfying the condition is found
    and default is not given or doesn't satisfy the condition.

    >>> first( (1,2,3), condition=lambda x: x % 2 == 0)
    2
    >>> first(range(3, 100))
    3
    >>> first( () )
    Traceback (most recent call last):
    ...
    StopIteration
    >>> first([], default=1)
    1
    >>> first([], default=1, condition=lambda x: x % 2 == 0)
    Traceback (most recent call last):
    ...
    StopIteration
    >>> first([1,3,5], default=1, condition=lambda x: x % 2 == 0)
    Traceback (most recent call last):
    ...
    StopIteration
    """

    try:
        return next(x for x in iterable if condition(x))
    except StopIteration:
        if default is not None and condition(default):
            return default
        else:
            raise
