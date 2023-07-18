#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 15:23:43 2023.

@author: placais

This module contains several utilities to evaluate the 'quality' of a fit. In
other words: How close a fixed linac is to a reference accelerator? Do the
fixed accelerator has valid settings, or do they raise any red flag?

"""
from typing import Any
import logging

import numpy as np
import pandas as pd

from core.elements import _Element
from beam_calculation.output import SimulationOutput
from optimisation.fault import Fault
from util import helper


# TODO return values as string with their units
def compute_differences_between_simulation_outputs(
        simulation_outputs: tuple[SimulationOutput],
        quantities_to_evaluate: tuple[str],
        *tests: str,
        **kwargs: Any,
) -> pd.DataFrame:
    """Evaluate difference on several quantities between ref and fix linac."""

    test_outputs = {'Qty': quantities_to_evaluate}
    for test in tests:
        output = DIFFERENCE_TESTERS[test](*simulation_outputs,
                                          *quantities_to_evaluate,
                                          **kwargs)
        test_outputs.update(output)

    df_eval = pd.DataFrame(columns=list(test_outputs.keys()))
    df_eval['Qty'] = quantities_to_evaluate
    for evaluated_test, evaluated_quantities in test_outputs.items():
        df_eval[evaluated_test] = evaluated_quantities
    logging.info(helper.pd_output(df_eval, header='Fit evaluation'))
    return df_eval


def _diff_at_elements_exits(ref_simulation_output: SimulationOutput,
                            fix_simulation_output: SimulationOutput,
                            *quantities_to_evaluate: str, **kwargs: Any,
                            ) -> dict[str, list[float]]:
    """
    Evaluate error of `quantities_to_evaluate` at some _Elements exits.

    Error is relative error in percents between `ref` and `fix`. Only
    exception is for 'mismatch_factor' key (not multiplied by 100).

    Parameters
    ----------
    ref_simulation_output : SimulationOutput
        Reference SimulationOutput.
    fix_simulation_output : SimulationOutput
        Fixed SimulationOutput.
    *quantities_to_evaluate : str
        Tuple containing the keys corresponding to the desired quantities. Must
        work with the SimulationOutput.get method.
    **kwargs: Any
        Here we pass `faults` and `additional_elts` keys to get the list of
        _Elements where quantities are evaluated.

    Returns
    -------
    evaluated_quantities : dict[str, list[float]]
        Contains all the `quantities_to_evaluate` at the end of the
        compensation zone of every `faults` and at the exit of every element of
        `additional_elts`. Keys are headers for better output.

    """
    evaluation_elements = _get_diff_at_elements_exits_elements(**kwargs)
    evaluated_quantities = {header: [] for header in evaluation_elements}

    for header, elt in evaluation_elements.items():
        for quantity in quantities_to_evaluate:
            fix = fix_simulation_output.get(quantity, elt=elt, pos='out')
            if quantity == 'mismatch_factor':
                evaluated_quantities[header].append(fix)
                continue

            ref = ref_simulation_output.get(quantity, elt=elt, pos='out')
            evaluated_quantities[header].append(1e2 * (ref - fix) / ref)
    return evaluated_quantities


def _get_diff_at_elements_exits_elements(faults: tuple[Fault] = (),
                                         elts: tuple[_Element] = (),
                                         additional_elts: tuple[_Element] = (),
                                         **kwargs) -> dict[str, _Element]:
    """
    Set the _Elements for the `_diff_at_elements_exits` test.

    Parameters
    ----------
    faults : tuple[Fault], optional
        List of Faults contained in the FaultScenario that called this routine.
        The default is None, which raises an error.
    additional_elts : tuple[_Element], optional
        List of additional _Element where `quantities_to_evaluate` should be
        calculated. The default is empty tuple ().
    **kwargs : Any
        Not used.

    Returns
    -------
    evaluation_elements : dict[str, _Element]
        Holds all _Elements where `quantities_to_evaluate` will be calculated.

    """
    if len(faults) == 0:
        logging.error("Need to pass the list of Faults for this test.")
        return {}

    if len(elts) == 0:
        logging.error("Need to pass the list of _Elements for this test.")
        return {}

    evaluation_elements = {"end comp zone": fault.elts[-1] for fault in faults}
    evaluation_elements['end linac'] = elts[-1]
    for elt in additional_elts:
        evaluation_elements["user-defined"] = elt
    evaluation_elements = {header + f"\n({elt=})": elt
                           for header, elt in evaluation_elements.items()}
    return evaluation_elements


# FIXME: not RMS error. Plus: maybe there is much more relatable crits.
def _diff_over_full_accelerator(ref_simulation_output: SimulationOutput,
                                fix_simulation_output: SimulationOutput,
                                *quantities_to_evaluate: str, **kwargs: Any,
                                ) -> dict[str, list[float]]:
    """
    Evaluate sum of RMS errors over full linac of `quantities_to_evaluate`.

    Parameters
    ----------
    ref_simulation_output : SimulationOutput
        Reference SimulationOutput.
    fix_simulation_output : SimulationOutput
        Fixed SimulationOutput.
    *quantities_to_evaluate : str
        Tuple containing the keys corresponding to the desired quantities. Must
        work with the SimulationOutput.get method.
    **kwargs: Any
        Here we pass `faults` and `additional_elts` keys to get the list of
        _Elements where quantities are evaluated.

    Returns
    -------
    evaluated_quantities : dict[str, list[float]]
        Contains all the `quantities_to_evaluate` sumed over the linac. Key is
        header for better output.

    """
    header = 'sum over linac'
    evaluated_quantities = {header: []}

    for quantity in quantities_to_evaluate:
        fix = fix_simulation_output.get(quantity)
        if quantity == 'mismatch_factor':
            evaluated_quantities[header].append(np.sum(fix))
            continue

        ref = ref_simulation_output.get(quantity)
        ref[ref == 0.] = np.NaN
        error = np.nansum(np.sqrt(((ref - fix) / ref)**2))
        evaluated_quantities[header].append(error)
    return evaluated_quantities


DIFFERENCE_TESTERS = {
    'elements exits': _diff_at_elements_exits,
    'over accelerator': _diff_over_full_accelerator
}
