"""Test the ``wtf`` (what to fit) key of the config file.

.. todo::
    Specific test for every optimisation method? For now, just trust the user.

"""

import logging

from config.helper import check_type

IMPLEMENTED_STRATEGIES = (
    "k out of n",
    "manual",
    "l neighboring lattices",
    "global",
    "global downstream",
)  #:
IMPLEMENTED_OBJECTIVE_PRESETS = (
    "EnergyPhaseMismatch",
    "simple_ADS",  # equivalent
    "EnergyMismatch",
    "rephased_ADS",  # equivalent
    "EnergySyncPhaseMismatch",
    "sync_phase_as_objective_ADS",  # equivalent
    "experimental",
)  #:
IMPLEMENTED_OPTIMISATION_ALGORITHMS = (
    "least_squares",
    "least_squares_penalty",
    "nsga",
    "downhill_simplex",
    "downhill_simplex_penalty",
    "nelder_mead",
    "differential_evolution",
    "explorator",
    "experimental",
)  #:


def test(
    id_nature: str,
    strategy: str,
    objective_preset: str,
    optimisation_algorithm: str,
    tie_politics: str = "upstream first",
    idx: str = "",
    shift: int = 0,
    **wtf_kw,
) -> None:
    """Test the ``wtf`` ``.toml`` entries."""
    assert id_nature in ("cavity", "element", "name")
    if idx:
        logging.error("Deprecated, use 'id_nature' instead.")
    assert strategy in IMPLEMENTED_STRATEGIES
    strategy_testers = {
        "k out of n": _test_k_out_of_n,
        "manual": _test_manual,
        "l neighboring lattices": _test_l_neighboring_lattices,
        "global": _test_global,
        "global downstream": _test_global_downstream,
    }
    strategy_testers[strategy](**wtf_kw)

    assert objective_preset in IMPLEMENTED_OBJECTIVE_PRESETS
    assert optimisation_algorithm in IMPLEMENTED_OPTIMISATION_ALGORITHMS
    _test_cavity_selection(tie_politics, shift)


def _test_cavity_selection(tie_politics: str, shift: int) -> None:
    """Test the keywords that alter the selection of compensating cavities."""
    allowed = ("upstream first", "downstream first")
    assert tie_politics in allowed, f"{tie_politics = } but {allowed = }"

    if (shift > 0 and tie_politics == "upstream first") or (
        shift < 0 and tie_politics == "downstream first"
    ):
        logging.warning(
            f"{tie_politics = } is inconsistent with {shift = }. Is this what "
            "you want? You should double check the compensating cavities for "
            "the fault you are studying."
        )


def _test_k_out_of_n(k: int, **wtf_kw) -> None:
    """Test that the k out of n method can work."""
    check_type(int, "wtf", k)


def _test_manual(
    failed: list[list[list[int]]],
    compensating_manual: list[list[list[int]]],
    **wtf_kw,
) -> None:
    """Test that the manual method can work."""
    check_type(list, "wtf", failed, compensating_manual)
    assert len(failed) == len(compensating_manual), (
        "Discrepancy between the number of FaultScenarios and the number of "
        "corresponding list of compensating cavities. In other words: "
        "'failed[i]' and 'compensating_manual[i]' entries must have the same "
        "number of elements."
    )

    for scenarios, grouped_compensating_cavities in zip(
        failed, compensating_manual
    ):
        check_type(list, "wtf", scenarios, grouped_compensating_cavities)
        assert len(scenarios) == len(grouped_compensating_cavities), (
            "In a FaultScenario, discrepancy between the number of fault "
            "groups and group of compensating cavities. In other words: "
            "'failed[i][j]' and 'compensating_manual[i][j]' entries must have "
            "the same number of elements."
        )
        for failure in scenarios:
            check_type(int, "wtf", failure)
        for compensating_cavity in grouped_compensating_cavities:
            check_type(int, "wtf", compensating_cavity)


def _test_l_neighboring_lattices(
    l: int, min_number_of_cavities_in_lattice: int = 1, **wtf_kw
) -> None:
    check_type(int, "wtf", l)
    check_type(int, "wtf", min_number_of_cavities_in_lattice)


def _test_global(**wtf_kw) -> None:
    """Test if global compensation will work."""
    logging.error("Not tested.")


def _test_global_downstream(**wtf_kw) -> None:
    """Test if global compensation with only downstream cavities will work."""
    logging.error("Not tested.")
