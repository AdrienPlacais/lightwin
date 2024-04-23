"""Define helper function to ease lists manipulation in :mod:``.strategy``."""

import itertools
import math
from collections.abc import Callable, Sequence
from functools import partial


def _distance_to_ref[
    T
](
    element: T,
    elts_of_interest: Sequence[T],
    elements: Sequence[T],
    tie_politics: str,
) -> tuple[int, int]:
    """Give distance between the two elements in ``elements``.

    Parameters
    ----------
    element : T
        First object from which you want distance. Often, an :class:`.Element`
        of a lattice that will potentially be used for compensation.
    elts_of_interest : Sequence[T]
        Second object or list of object from which you want distance. Often, a
        list of failed :class:`.Element` or a list of lattices with a fault.
    elements : Sequence[T]
        All the elements/lattices/sections.
    tie_politics : {'upstream first', 'downstream first'}
        When two elements have the same position, will you want to have the
        upstream or the downstream first?

    Returns
    -------
    lowest_distances : int
        Index-distance between ``element`` and closest element of
        ``elts_of_interest``. Will be used as a primary sorting key.
    index : int
        Index of ``element``. Will be used as a secondary index key, to sort
        ties in distance.

    """
    index = elements.index(element)
    distances = (abs(index - elements.index(elt)) for elt in elts_of_interest)
    lowest_distance = min(distances)

    if tie_politics == "upstream first":
        return lowest_distance, index
    if tie_politics == "downstream first":
        return lowest_distance, -index
    raise IOError(f"{tie_politics = } not understood.")


def sort_by_position[
    T
](
    elements: Sequence[T],
    elts_of_interest: Sequence[T],
    tie_politics: str = "upstream first",
) -> Sequence[T]:
    """Sort given list by how far its elements are from ``elements[idx]``."""
    sorter = partial(
        _distance_to_ref,
        elts_of_interest=elts_of_interest,
        elements=elements,
        tie_politics=tie_politics,
    )
    return sorted(elements, key=lambda element: sorter(element))


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
