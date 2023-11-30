#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tools to perform post-treatments/evaluations of the settings found by LightWin.

The presets that are listed here are defined and documented in
:mod:`evaluator.simulation_output_evaluator_presets`. This is where you should
add you own presets.

"""
import logging
import configparser

evaluators = (
        "optimisation time",
        "no power loss",
        "transverse eps_x shall not grow too much",
        "transverse eps_y shall not grow too much",
        "longitudinal eps shall not grow too much",
        "max of 99percent transverse eps_x shall not be too high",
        "max of 99percent transverse eps_y shall not be too high",
        "max of 99percent longitudinal eps shall not be too high",
        "longitudinal eps at end",
        "transverse eps at end",
        "mismatch factor at end",
        "transverse mismatch factor at end",
        )  #:


def test(c_evaluators: configparser.SectionProxy) -> None:
    """Test that the provided evaluations can be performed."""
    passed = True

    implemented = ['beam_calc_post']

    for key in c_evaluators.keys():
        if key not in implemented:
            logging.error(f"The evaluators {key} is not implemented. "
                          f"Authorized values are: {implemented}.")
            passed = False

        evaluations = c_evaluators.gettuplestr(key)
        for evaluation in evaluations:
            if evaluation not in evaluators:
                logging.error(f"The evaluator {evaluation} is not implemented."
                              f" Authorized values are: {evaluators}."
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
