"""Define helper function to ease lists manipulation in :mod:``.strategy``.

.. note::
    If you are unsure about how a function works, check out the implementation
    of the tests in ``lightwin/tests/test_failure/test_helper.py``.

"""

import itertools
import math
from collections.abc import Callable, Collection, Sequence
from functools import partial


def _distance_to_ref[
    T
](
    element: T,
    failed: Sequence[T],
    all_elements: Sequence[T],
    tie_politics: str,
    shift: int = 0,
) -> tuple[int, int]:
    """Give distance between ``element`` and closest of ``failed``.

    Parameters
    ----------
    element : T
        First object from which you want distance. Often, an :class:`.Element`
        of a lattice that will potentially be used for compensation.
    failed : Sequence[T]
        Second object or list of object from which you want distance. Often, a
        list of failed :class:`.Element` or a list of lattices with a fault.
    all_elements : Sequence[T]
        All the elements/lattices/sections.
    tie_politics : {'upstream first', 'downstream first'}
        When two elements have the same position, will you want to have the
        upstream or the downstream first?
    shift : int, optional
        Distance increase for downstream elements (``shift < 0``) or upstream
        elements (``shift > 0``). Used to have a window of compensating
        cavities which is not centered around the failed elements. The default
        is 0.

    Returns
    -------
    lowest_distance : int
        Index-distance between ``element`` and closest element of ``failed``.
        Will be used as a primary sorting key.
    index : int
        Index of ``element``. Will be used as a secondary index key, to sort
        ties in distance.

    """
    index = all_elements.index(element)
    distances = (
        abs(index - (failure_index := all_elements.index(failed_element)))
        + _penalty(index, failure_index, shift)
        for failed_element in failed
    )
    lowest_distance = min(distances)

    if tie_politics == "upstream first":
        return lowest_distance, index
    if tie_politics == "downstream first":
        return lowest_distance, -index
    raise IOError(f"{tie_politics = } not understood.")


def _penalty(index: int, failure_index: int, shift: int) -> int:
    """Give the distance penalty.

    .. note::
        If ``shift > 0``, upstream elements are penalized.
        If ``shift < 0``, downstream elements are penalized.

    """
    if index == failure_index:
        return 0
    if (failure_index < index) is not (shift < 0):
        return 0
    return abs(shift)


def sort_by_position[
    T
](
    all_elements: Sequence[T],
    failed: Sequence[T],
    tie_politics: str = "upstream first",
    shift: int = 0,
) -> Sequence[T]:
    """Sort given list by how far its elements are from ``elements[idx]``.

    We go across every element in ``all_elements`` and get their index-distance
    to the closest element of ``failed``. We sort ``all_elements`` by this
    distance. When there is a tie, we put the more upstream or the more
    downstream cavity first according to ``tie_politics``.

    Parameters
    ----------
    failed : Sequence[T]
        Second object or list of object from which you want distance. Often, a
        list of failed :class:`.Element` or a list of lattices with a fault.
    all_elements : Sequence[T]
        All the elements/lattices/sections.
    tie_politics : {'upstream first', 'downstream first'}
        When two elements have the same position, will you want to have the
        upstream or the downstream first?
    shift : int, optional
        Distance increase for downstream elements (``shift < 0``) or upstream
        elements (``shift > 0``). Used to have a window of compensating
        cavities which is not centered around the failed elements. Useful when
        upstream cavities have more important power margins, or when you want
        more downstream cavities because a full cryomodule is down. The default
        is 0.

    """
    sorter = partial(
        _distance_to_ref,
        failed=failed,
        all_elements=all_elements,
        tie_politics=tie_politics,
        shift=shift,
    )
    return sorted(all_elements, key=lambda element: sorter(element))


def remove_lists_with_less_than_n_elements[
    T
](elements: list[Sequence[T]], minimum_size: int = 1) -> list[Sequence[T]]:
    """Return a list where objects have a minimum length of ``minimum_size``."""
    out = [x for x in elements if len(x) >= minimum_size]
    return out


def gather[
    T
](
    failed_elements: list[T],
    fun_sort: Callable[[Sequence[T] | Sequence[Sequence[T]]], list[T]],
) -> tuple[list[list[T]], list[list[T]]]:
    """Gather faults to be fixed together and associated compensating cav.

    Parameters
    ----------
    failed_elements : list[T]
        Holds ungathered failed cavities.
    fun_sort : Callable[[Sequence[T] | Sequence[Sequence[T]]], list[T]]
        Takes in a list or a list of list of failed cavities, returns the list
        or list of list of altered cavities (failed + compensating).

    Returns
    -------
    failed_gathered : list[list[T]]
        Failures, gathered by faults that require the same compensating
        cavities.
    compensating_gathered : list[list[T]]
        Corresponding compensating cavities.

    """
    r_comb = 2

    flag_gathered = False
    altered_gathered: list[list[T]] = []
    failed_gathered = [[failed] for failed in failed_elements]
    while not flag_gathered:
        # List of list of corresp. compensating cavities
        altered_gathered = [
            fun_sort(failed_elements=failed) for failed in failed_gathered
        ]

        # Set a counter to exit the 'for' loop when all faults are gathered
        i = 0
        n_combinations = len(altered_gathered)
        if n_combinations <= 1:
            flag_gathered = True
            break
        i_max = int(
            math.factorial(n_combinations)
            / (
                math.factorial(r_comb)
                * math.factorial(n_combinations - r_comb)
            )
        )

        # Now we look every list of required compensating cavities, and
        # look for faults that require the same compensating cavities
        for (idx1, altered1), (idx2, altered2) in itertools.combinations(
            enumerate(altered_gathered), r_comb
        ):
            i += 1
            common = list(set(altered1) & set(altered2))
            # If at least one cavity on common, gather the two
            # corresponding fault and restart the whole process
            if len(common) > 0:
                failed_gathered[idx1].extend(failed_gathered.pop(idx2))
                altered_gathered[idx1].extend(altered_gathered.pop(idx2))
                break

            # If we reached this point, it means that there is no list of
            # faults that share compensating cavities.
            if i == i_max:
                flag_gathered = True

    compensating_gathered = [
        list(filter(lambda cavity: cavity not in failed_elements, sublist))
        for sublist in altered_gathered
    ]
    return failed_gathered, compensating_gathered


def nested_containing_desired[
    T
](
    nested: Collection[Sequence[T]],
    desired_elements: Collection[T],
) -> list[
    Sequence[T]
]:
    """Return collections of ``nested`` containing some ``desired_elements``.

    Example
    -------
    nested_containing_desired(ListOfElements.by_lattice,
                              failed_elements
                              ) -> lattices_with_a_failure

    """
    nested_with_desired_elements = [
        x for x in nested if not set(desired_elements).isdisjoint(x)
    ]
    return nested_with_desired_elements
