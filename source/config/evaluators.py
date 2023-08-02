#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 09:21:50 2023.

@author: placais

Configuration of the tools to perform the post-treatments/evaluations of the
settings found by LightWin.

"""
import logging
import configparser


def test(c_evaluators: configparser.SectionProxy) -> None:
    """Test that the provided evaluations can be performed."""
    passed = True

    implemented = ['beam_calc_post']
    implemented_presets = (
        "no power loss",
        "longitudinal eps shall not grow too much",
        "max of eps shall not be too high",
        "longitudinal eps at end",
        "mismatch factor at end"
    )

    for key in c_evaluators.keys():
        if key not in implemented:
            logging.error(f"The evaluators {key} is not implemented. "
                          f"Authorized values are: {implemented}.")
            passed = False

        evaluations = c_evaluators.gettuplestr(key)
        for evaluation in evaluations:
            if evaluation not in implemented_presets:
                logging.error(f"The evaluator {evaluation} is not implemented."
                              f" Authorized values are: {implemented_presets}."
                              "Add your preset in evaluator.simulation_output_"
                              "evaluator_presets.py. And also in config.evalua"
                              "tors.py, these two have issues communicating.")
                passed = False

    if not passed:
        raise IOError("Error treating the evaluator parameters.")

    logging.info(f"files parameters {c_evaluators.name} tested with success.")


def config_to_dict(c_evaluators: configparser.SectionProxy) -> dict:
    """Save evaluators info into a dict."""
    evaluators = {}
    for key in c_evaluators.keys():
        evaluators[key] = c_evaluators.gettuplestr(key)
    return evaluators
