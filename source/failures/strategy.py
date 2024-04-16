#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Define the function related to the ``strategy`` key of ``wtf``.

In particular, it answers the question:
**Given this set of faults, which compensating cavities will be used?**

.. note::
    In this module, the indexes are **cavity** indexes, not element.

.. note::
    In order to add a compensation strategy, you must give it an
    ``AlteredCavityGiver`` function, add it to the ``altered_cavities_givers``
    dict, and also to the list of supported strategies in
    :mod:`config.failures.strategy` module.

"""
import itertools
import logging
import math
from typing import Any, Callable, TypeAlias

import numpy as np

from core.accelerator.accelerator import Accelerator
from core.list_of_elements.helper import filter_cav
from util.helper import flatten

FACT = math.factorial

# The base "template" that each strategy has to respect
# Takes in an Accelerator, a list of failed cavities, some **kwargs, and
# returns a list of altered (failed or compensating) cavities.
AlteredCavityGiver: TypeAlias = Callable[
    [Accelerator, list[int], Any | None], list[int]
]


def sort_and_gather_faults(
    fix: Accelerator,
    fault_idx: list[int] | list[list[int]],
    idx: str,
    strategy: str,
    comp_idx: list[list[int]] | None = None,
    **wtf: Any,
) -> tuple[list[list[int]], list[list[int]]]:
    """Link faulty cavities with their compensating cavities.

    If two faults need the same compensating cavities, they are gathered, their
    compensating cavities are put in common and they will be fixed together.

    Parameters
    ----------
    fix : Accelerator
        Broken linac to fix.
    fault_idx : list[int] | list[list[int]]
        The indexes of the cavities to fix.
    idx : {'element', 'cavity'}
        Entry for ``idx`` in the ``.ini`` file. Tells if the entry
        ``failed = 10`` means that the 10th element is broken, or that the 10th
        cavity is broken.
    strategy : str
        Entry for ``strategy`` in the ``.ini`` file.
    comp_idx : list[list[int]] | None, optional
        List of compensating cavities. Only used with manual strategy. The
        default is None.
    wtf : Any
        The wtf dictionary from the ``.ini`` config file.

    Returns
    -------
    gathered_fault_idx : list[list[int]]
        The failed cavities. Cavities to be fixed during the same optimisation
        run are grouped together.
    gathered_comp_idx : list[list[int]]
        The compensating cavities. Length of the outer list if the same as
        ``gathered_fault_idx``.

    """
    for my_list in [fault_idx, comp_idx]:
        assert _only_field_maps(fix, my_list, idx=idx)

        if idx == "element":
            my_list = _to_cavity_idx(fix, my_list)

    if strategy == "manual":
        gathered_comp_idx, gathered_comp_idx = _manual(fault_idx, comp_idx)
        return gathered_comp_idx, gathered_comp_idx

    gathered_fault_idx, gathered_comp_idx = _gather(
        fix, fault_idx, strategy, **wtf
    )
    return gathered_fault_idx, gathered_comp_idx


def _gather(
    fix: Accelerator, fault_idx: list[int], strategy: str, **wtf: Any
) -> tuple[list[list[int]], list[list[int]]]:
    """Gather faults to be fixed together and associated compensating cav."""
    fun_sort = altered_cavities_givers[strategy]
    r_comb = 2

    flag_gathered = False
    gathered_faults = [fault_idx]
    while not flag_gathered:
        # List of list of corresp. compensating cavities
        gathered_comp = [
            fun_sort(fix, faults, **wtf) for faults in gathered_faults
        ]

        # Set a counter to exit the 'for' loop when all faults are gathered
        i = 0
        n_combinations = len(gathered_comp)
        if n_combinations <= 1:
            flag_gathered = True
            break
        i_max = int(
            FACT(n_combinations)
            / (FACT(r_comb) * FACT(n_combinations - r_comb))
        )

        # Now we look every list of required compensating cavities, and
        # look for faults that require the same compensating cavities
        for (idx1, l_comp1), (idx2, l_comp2) in itertools.combinations(
            enumerate(gathered_comp), r_comb
        ):
            i += 1
            common = list(set(l_comp1) & set(l_comp2))
            # If at least one cavity on common, gather the two
            # corresponding fault and restart the whole process
            if len(common) > 0:
                gathered_faults[idx1].extend(gathered_faults.pop(idx2))
                gathered_comp[idx1].extend(gathered_comp.pop(idx2))
                break

            # If we reached this point, it means that there is no list of
            # faults that share compensating cavities.
            if i == i_max:
                flag_gathered = True

    gathered_comp = [
        list(filter(lambda idx: idx not in fault_idx, sublist))
        for sublist in gathered_comp
    ]
    return gathered_faults, gathered_comp


def _manual(
    fault_idx: list[list[int]] | list[int],
    comp_idx: list[list[int]] | None,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return the altered cavities, as desired by user.

    This function is different from the other strategy func, as all
    responsibilities rely on the user. Here, he/she is the one that select
    which compensating cavities will be used for every failure.
    In the ``.ini`` file, first line of ``manual list`` entry will compensate
    first line of ``failed``, etc.

    Example
    -------
    We consider that ``idx = cavity``.

    .. code-block:: ini

        failed =
            10, 11,
            27,
            10, 11 | 27
        manual list =
            8, 9, 12, 13
            25, 26, 28, 29
            8, 9, 12, 13 | 25, 26, 28, 29

    In this scenario, three simulations will be runned.

        - Cavities 10 and 11 down, compensated with 8, 9, 12, 13.
        - Cavity 27 down, compensated with 25, 26, 28, 29.
        - Cavities 10, 11 are down, compensated with 8, 9, 12, 13. Once it is
          done, beam is propagated up to the broken cavity 27, compensated with
          25, 26, 28, 29. Compensation settings corresponding to cavity 27
          error may be different from simulation #2, especially if the nominal
          beam parameters are not retrieved after the errors 10 and 11.


    Parameters
    ----------
    fault_idx : list[list[int]]
        fault_idx
    comp_idx : list[list[int]]
        comp_idx

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]

    """
    assert all(isinstance(idx, list) for idx in fault_idx)
    assert comp_idx is not None
    return fault_idx, comp_idx


def _k_out_of_n(
    lin: Accelerator, fault_idx: list[int], k: int, **wtf: Any
) -> list[int]:
    """Select the cavities neighboring the failed one(s).

    This is the 'k out of n' strategy as defined by Biarrotte [1]_ and used by
    Yee-Rendon et al. [2]_. `k` compensating cavities per failure. Nearby
    broken cavities are automatically gathered and fixed together.

    References
    ----------
    .. [1] J. L. Biarrotte, "Reliability and fault tolerance in the European
        ADS project", in Proceedings of CERN Accelerator School on High Power
        Hadron Machines, CAS- 2011, Bilbao, Spain, 2011 (CERN, Geneva,
        Switzerland, 2013), pp. 481–494. arXiv:1307.8304

    .. [2] B. Yee-Rendon, Y. Kondo, J. Tamura, K. Nakano, F. Maekawa, and S.-i.
        Meigo, "Beam dynamics studies for fast beam trip recovery of the Japan
        Atomic Energy Agency accelerator-driven subcritical system," Physical
        Review Accelerators and Beams, vol. 25, no. 8, p. 080 101, 2022.
        doi:10.1103/PhysRevAccelBeams.25.080101

    Parameters
    ----------
    lin : Accelerator
        Accelerator under study.
    fault_idx : list[int]
        Cavity index of the failures.
    k : int
        Number of compensating cavities per failed cavity.
    wtf : dict
        Dictionary defined in the LightWin ``.ini`` file.

    Returns
    -------
    idx_altered : list[int]
        Cavity index of compensating and failed cavities.

    """
    # "altered" means compensating or failed cavity
    n_altered_cav = len(fault_idx) * (k + 1)

    # List of distances between the failed cavities and the other ones
    distances = []
    for idx in fault_idx:
        distance = np.array(
            [idx - lin.l_cav.index(cav) for cav in lin.l_cav], dtype=np.float64
        )
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

    idx_altered = np.lexsort((sort_bis, distance))[:n_altered_cav]
    idx_altered.sort()
    idx_altered = list(idx_altered)
    return idx_altered


def _l_neighboring_lattices(
    lin: Accelerator, fault_idx: list[int], l: int, **wtf: Any
) -> list[int]:
    """Select full lattices neighboring the failed cavities.

    Every fault will be compensated by ``l`` full lattices, direct neighbors of
    the errors [1, 2]. You must provide l, which must be even.

    .. todo::
        Handle non-even ``l``, plus a way to select if the extra lattice should
        be before or after the failure.

    References
    ----------
    .. [1] F. Bouly, J.-L. Biarrotte, and D. Uriot, "Fault tolerance and
        consequences in the MYRRHA superconducting LINAC," in Proc. 27th Linear
        Accelerator Conf. (LINAC14). Geneva, Switzerland: JACoW Publishing,
        Geneva, Switzerland, 2014.
    .. [2] A. Plaçais and F. Bouly, "Cavity Failure Compensation Strategies in
        Superconducting Linacs," in Proceedings of LINAC2022, 2022, pp.
        552–555. doi:10.18429/JACoW-LINAC2022-TUPORI04

    Parameters
    ----------
    lin : Accelerator
        Accelerator under study.
    fault_idx : list[int]
        Cavity indexes of the failures.
    l : int
        The (even) number of lattices per failed cavity.
    wtf : dict
        Dictionary defined in the LightWin ``.ini`` file.

    Returns
    -------
    idx_altered : list[int]
        Indexes of failed and compensating cavities.

    """
    # For this routine, we use list of faulty cavities instead of list of idx
    fault_cav = [lin.l_cav[idx] for idx in fault_idx]

    cavities_by_lattice = [
        filter_cav(lattice) for lattice in lin.elts.by_lattice
    ]
    lattices_with_a_fault = [
        lattice
        for cav in fault_cav
        for lattice in cavities_by_lattice
        if cav in lattice
    ]
    faulty_latt_idx = [
        cavities_by_lattice.index(lattice) for lattice in lattices_with_a_fault
    ]

    half_n_latt = int(len(fault_idx) * l / 2)
    lattices_with_a_compensating_cavity = cavities_by_lattice[
        faulty_latt_idx[0]
        - half_n_latt : faulty_latt_idx[-1]
        + half_n_latt
        + 1
    ]

    # List of cavities in the compensating lattices
    idx_compensating = [
        lin.l_cav.index(cav)
        for lattice in lattices_with_a_compensating_cavity
        for cav in lattice
    ]
    return idx_compensating


def _global(lin: Accelerator, fault_idx: list[int], **wtf: Any) -> list[int]:
    """Select all the cavities of the linac."""
    cavities = lin.l_cav
    idx_altered = [idx for idx in range(len(cavities))]
    return idx_altered


def _global_downstream(
    lin: Accelerator, fault_idx: list[int], **wtf: dict
) -> list[int]:
    """Select all the cavities after the first failed cavity."""
    altered_cavities = lin.l_cav[min(fault_idx) :]
    idx_altered = [lin.l_cav.index(cav) for cav in altered_cavities]
    return idx_altered


def _only_field_maps(
    lin: Accelerator, indexes: list[int] | list[list[int]] | None, idx: str
) -> bool:
    """Check that all the required elements are cavities."""
    if indexes is None:
        return True

    elts = lin.elts
    if idx == "cavity":
        elts = lin.l_cav

    natures = set([elts[i].get("nature") for i in flatten(indexes)])
    if natures != {"FIELD_MAP"}:
        logging.error("Some elements are not cavities.")
        return False
    return True


def _to_cavity_idx(
    lin: Accelerator, indexes: list[int] | list[list[int]] | None
) -> list[int] | list[list[int]] | None:
    """
    Convert ``i``-th element to ``k``-th cavity.

    Works with list of indexes (ungathered) and list of list of indexes
    (gathered, which is when ``method = 'manual'``).

    """
    if indexes is None:
        return None

    set_types = {type(idx) for idx in indexes}
    list_in = lin.elts
    list_out = lin.l_cav

    if set_types == {int}:
        elts = [list_in[i] for i in indexes]
        indexes = [list_out.index(elt) for elt in elts]
        return indexes

    if set_types == {list}:
        grouped_elts = [[list_in[i] for i in idx] for idx in indexes]
        grouped_indexes = [
            [list_out.index(elt) for elt in elts] for elts in grouped_elts
        ]
        return grouped_indexes

    logging.error(f"{indexes} data type was not recognized.")
    return None


altered_cavities_givers: dict[str, Callable]
altered_cavities_givers = {
    "k out of n": _k_out_of_n,
    "l neighboring lattices": _l_neighboring_lattices,
    "global": _global,
    "global downstream": _global_downstream,
}
