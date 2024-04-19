"""Define helper function to ease lists manipulation in :mod:``.strategy``."""

from collections.abc import Sequence
from functools import partial


def _distance_to_ref(
    item: object,
    main_item: object,
    objects: Sequence,
    tie_politics: str,
) -> tuple[int, int]:
    """Give distance between the two items in ``items``.

    Parameters
    ----------
    item, main_item : object
        Objects between which you want the distance.
    objects : Sequence[object]
        All the items.
    tie_politics : {'upstream first', 'downstream first'}
        When two items have the same position, will you want to have the
        upstream or the downstream first?

    Returns
    -------
    distance : int
        Index-distance between ``item`` and ``main_item``. Will be used as a
        primary sorting key.
    index : int
        Index of ``item``. Will be used as a secondary index key, to sort ties
        in distance.

    """
    index = objects.index(item)
    distance = abs(index - objects.index(main_item))

    if tie_politics == "upstream first":
        return distance, index
    if tie_politics == "downstream first":
        return distance, -index
    raise IOError(f"{tie_politics = } not understood.")


def sort_by_position(
    objects: Sequence[object],
    main_item: object,
    tie_politics: str = "upstream first",
) -> Sequence[object]:
    """Sort given list by how far its items are from ``objects[idx]``."""
    sorter = partial(
        _distance_to_ref,
        main_item=main_item,
        objects=objects,
        tie_politics=tie_politics,
    )
    return sorted(objects, key=lambda item: sorter(item))


def _k_out_of_n(
    sorted_by_position: Sequence[object],
    k: int,
    n: int,
) -> Sequence[object]:
    """Return ``k`` closest items to provided ``main_items``."""
    return sorted_by_position[n : n + k]


def whole_k_out_of_n(
    objects: Sequence[object],
    main_items: Sequence[object],
    k: int,
    tie_politics: str,
) -> Sequence[object]:
    """Whole process."""
    n = len(main_items)
    if n > 1:
        raise NotImplementedError

    print(objects)
    for item in main_items:
        sorted_by_position = sort_by_position(
            objects, item, tie_politics=tie_politics
        )
        compensating = _k_out_of_n(sorted_by_position, k, n)
        print(f"{compensating} with {k = } and {tie_politics = }")
        return compensating
    raise IOError
