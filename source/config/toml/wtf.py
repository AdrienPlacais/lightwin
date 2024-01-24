#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test the ``wtf`` (what to fit) key of the config file.

.. todo::
    Specific test for every optimisation method? For now, just trust the user.

"""
import logging

from config.toml.helper import check_type


IMPLEMENTED_STRATEGIES = ('k out of n',
                          'manual',
                          'l neighboring lattices',
                          'global',
                          'global downstream',
                          )  #:
IMPLEMENTED_OBJECTIVE_PRESETS = ('simple_ADS',
                                 'sync_phase_as_objective_ADS',
                                 'experimental'
                                 )  #:
IMPLEMENTED_OPTIMISATION_ALGORITHMS = ('least_squares',
                                       'least_squares_penalty',
                                       'nsga',
                                       'downhill_simplex',
                                       'nelder_mead',
                                       'differential_evolution',
                                       'explorator',
                                       'experimental')  #:


def test(idx: str,
         # failed: list[list[int]] | list[list[list[int]]],
         strategy: str,
         objective_preset: str,
         optimisation_algorithm: str,
         phi_s_fit: bool | None = None,
         **wtf_kw,
         ) -> None:
    """Test the ``wtf`` ``.toml`` entries."""
    assert idx in ('cavity', 'element')
    # check_type(list, 'wtf', failed)
    # if len(failed) == 0:
    #     logging.warning("No fault was given.")

    assert strategy in IMPLEMENTED_STRATEGIES
    strategy_testers = {'k out of n': _test_k_out_of_n,
                        'manual': _test_manual,
                        'l neighboring lattices': _test_l_neighboring_lattices,
                        'global': _test_global,
                        'global downstream': _test_global_downstream,
                        }
    strategy_testers[strategy](**wtf_kw)

    assert objective_preset in IMPLEMENTED_OBJECTIVE_PRESETS
    assert optimisation_algorithm in IMPLEMENTED_OPTIMISATION_ALGORITHMS

    if phi_s_fit is None:
        logging.error("Please explicitly precise if you want to fit synch "
                      "phases or not (equivalent of SET_SYNCH_PHASE)."
                      "Setting default: False.")
        phi_s_fit = False


def _test_k_out_of_n(k: int, **wtf_kw) -> None:
    """Test that the k out of n method can work."""
    check_type(int, 'wtf', k)


def _test_manual(failed: list[list[list[int]]],
                 manual_list: list[list[list[int]]],
                 **wtf_kw) -> None:
    """Test that the manual method can work."""
    check_type(list, 'wtf', failed, manual_list)
    assert len(failed) == len(manual_list), (
        "Discrepancy between the number of FaultScenarios and "
        "the number of corresponding list of compensating "
        "cavities. In other words: 'failed[i]' and 'manual list[i]' "
        "entries must have the same number of elements.")

    for scenarios, grouped_compensating_cavities in zip(failed, manual_list):
        check_type(list, 'wtf', scenarios, grouped_compensating_cavities)
        assert len(scenarios) == len(grouped_compensating_cavities), (
            "In a FaultScenario, discrepancy between the number of fault "
            "groups and group of compensating cavities. In other words: "
            "'failed[i][j]' and 'manual_list[i][j]' entries must have the same"
            " number of elements.")
        for failure in scenarios:
            check_type(int, 'wtf', failure)
        for compensating_cavity in grouped_compensating_cavities:
            check_type(int, 'wtf', compensating_cavity)


def _test_l_neighboring_lattices(l: int, **wtf_kw) -> None:
    check_type(int, 'wtf', l)
    assert l % 2 == 0, f"{l = } should be even."


def _test_global(**wtf_kw) -> None:
    """Test if global compensation will work."""
    logging.error("Not really implemented in fact.")


def _test_global_downstream(**wtf_kw) -> None:
    """Test if global compensation with only downstream cavities will work."""
    logging.error("Not really implemented in fact.")
