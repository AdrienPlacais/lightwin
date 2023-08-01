#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:39:57 2023.

@author: placais

In this module we store predefined arguments to generate
`SimulationOutputEvaluator`s.

"""
import logging
from typing import Callable

from functools import partial
import numpy as np

from core.elements import _Element
from beam_calculation.output import SimulationOutput
from evaluator import post_treaters, testers
from util.dicts_output import markdown


# =============================================================================
# "static" presets
# =============================================================================
PRESETS = {
    # Legacy "fit quality"
    # Legacy "Fred tests"
    "no power loss": {
        'value_getter': lambda s: s.get('pow_lost'),
        'post_treaters': (partial(post_treaters.do_nothing, to_plot=True),),
        'tester': partial(testers.value_is, objective_value=0., to_plot=True),
        'markdown': markdown["pow_lost"],
        'descriptor': """Lost power shall be null.""",
        'plt_kwargs': {'fignum': 101,
                       'savefig': True},
    },
    "longitudinal eps shall not grow too much": {
        'value_getter': lambda s: s.get('eps_zdelta'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_zdelta',
                                                   elt='first', pos='in'),
        'post_treaters': (post_treaters.relative_difference,
                          partial(post_treaters.scale_by,
                                  scale=100., to_plot=True),
                          post_treaters.maximum),
        'tester': partial(testers.value_is_below,
                          upper_limit=20., to_plot=True),
        'markdown': r"$\Delta\epsilon_{z\delta} / \epsilon_{z\delta}$ "
                    + r"(ref $z=0$) [%]",
        'descriptor': """Longitudinal emittance should not grow by more than
                         20% along the linac.""",
        'plt_kwargs': {'fignum': 102,
                       'savefig': True},

    },
    "max of eps shall not be too high": {
        'value_getter': lambda s: s.get('eps_zdelta'),
        'ref_value_getter': lambda ref_s, s: np.max(ref_s.get('eps_zdelta')),
        'post_treaters': (post_treaters.maximum,
                          partial(post_treaters.relative_difference,
                                  replace_zeros_by_nan_in_ref=False,
                                  to_plot=True)),
        'tester': partial(testers.value_is_below,
                          upper_limit=30., to_plot=True),
        'markdown': r"$\frac{max(\epsilon_{z\delta}) - "
                    + r"max(\epsilon_{z\delta}^{ref}))}"
                    + r"{max(\epsilon_{z\delta}^{ref})}$",
        'descriptor': """The maximum of longitudinal emittance should not
                         exceed the nominal maximum of longitudinal emittance
                         by more than 30%.""",
        'plt_kwargs': {'fignum': 103,
                       'savefig': True},

    },
    # Legacy "Bruce tests"
    "longitudinal eps at end": {
        'value_getter': lambda s: s.get('eps_zdelta', elt='last', pos='out'),
        'ref_value_getter': lambda ref_s, s: ref_s.get('eps_zdelta',
                                                       elt='last', pos='out'),
        'post_treaters': (post_treaters.relative_difference,),
        'markdown': markdown['eps_zdelta'],
        'descriptor': """Relative difference of emittance in [z-delta] plane
                         between fixed and reference linacs."""
    },
    "mismatch factor at end": {
        'value_getter': lambda s: s.get('mismatch_factor',
                                        elt='last', pos='out'),
        'markdown': markdown['mismatch_factor'],
        'descriptor': """Mismatch factor at the end of the linac."""
    },
}


# =============================================================================
# Functions to generate presets
# =============================================================================
def presets_for_fault_scenario_rel_diff_at_some_element(
    quantity: str, elt: _Element | str,
    ref_simulation_output: SimulationOutput
) -> dict[str, Callable | int | str | tuple[Callable]]:
    """
    Create the settings to evaluate a difference @ some element exit.

    Used for `FaultScenario`s.

    """
    kwargs = {'elt': elt, 'pos': 'out', 'to_deg': False}

    base_dict = {
        'value_getter': lambda s: s.get(quantity, **kwargs),
        'ref_value_getter': lambda ref_s, s: ref_s.get(quantity, **kwargs),
        'ref_simulation_output': ref_simulation_output,
        'post_treaters': (post_treaters.relative_difference,
                          partial(post_treaters.scale_by, scale=100.)),
        'markdown': markdown[quantity].replace('deg', 'rad'),
        'descriptor': f"""Relative difference of {quantity} ({elt}) between
                          fixed and reference linacs."""
    }

    if 'mismatch' in quantity:
        base_dict['ref_value_getter'] = None
        base_dict['post_treaters'] = (post_treaters.do_nothing,)
        base_dict['descriptor'].replace(f"Relative difference of {quantity}",
                                        "Mismatch factor")

    return base_dict


def presets_for_fault_scenario_rms_over_full_linac(
    quantity: str, ref_simulation_output: SimulationOutput
) -> dict[str, Callable | int | str | tuple[Callable]]:
    """
    Create the settings to evaluate a RMS error over full linac.

    Used for `FaultScenario`s.

    """
    kwargs = {'to_deg': False}

    base_dict = {
        'value_getter': lambda s: s.get(quantity, **kwargs),
        'ref_value_getter': lambda ref_s, s: ref_s.get(quantity, **kwargs),
        'ref_simulation_output': ref_simulation_output,
        'post_treaters': (post_treaters.rms_error,),
        'markdown': markdown[quantity].replace('deg', 'rad'),
        'descriptor': f"""RMS error of {quantity} between fixed and reference
                          linacs."""
    }

    if 'mismatch' in quantity:
        base_dict['value_getter'] = lambda s: np.NaN
        base_dict['ref_value_getter'] = None
        base_dict['post_treaters'] = (post_treaters.do_nothing,)

    return base_dict
