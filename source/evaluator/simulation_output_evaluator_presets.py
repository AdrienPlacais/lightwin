#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
We define predefined arguments to generate :class:`SimulationOutputEvaluator`.

We defining your presets, do not forget to add them to
:mod:`config.evaluators`.

"""
from typing import Callable

from functools import partial
import numpy as np

from core.elements.element import Element
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from evaluator import post_treaters, testers
from util.dicts_output import markdown


# =============================================================================
# "static" presets
# =============================================================================
PRESETS = {
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
    "transverse eps_x shall not grow too much": {
        'value_getter': lambda s: s.get('eps_x'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_x',
                                                   elt='first', pos='in'),
        'post_treaters': (post_treaters.relative_difference,
                          partial(post_treaters.scale_by,
                                  scale=100., to_plot=True),
                          post_treaters.maximum),
        'tester': partial(testers.value_is_below,
                          upper_limit=20., to_plot=True),
        'markdown': r"$\Delta\epsilon_{xx'} / \epsilon_{xx'}$ (ref $z=0$) [%]",
        'descriptor': """Transverse emittance should not grow by more than
                         20% along the linac.""",
        'plt_kwargs': {'fignum': 110, 'savefig': True},

    },
    "transverse eps_y shall not grow too much": {
        'value_getter': lambda s: s.get('eps_y'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_y',
                                                   elt='first', pos='in'),
        'post_treaters': (post_treaters.relative_difference,
                          partial(post_treaters.scale_by,
                                  scale=100., to_plot=True),
                          post_treaters.maximum),
        'tester': partial(testers.value_is_below,
                          upper_limit=20., to_plot=True),
        'markdown': r"$\Delta\epsilon_{yy'} / \epsilon_{yy'}$ (ref $z=0$) [%]",
        'descriptor': """Transverse emittance should not grow by more than
                         20% along the linac.""",
        'plt_kwargs': {'fignum': 111, 'savefig': True},

    },
    "longitudinal eps shall not grow too much": {
        'value_getter': lambda s: s.get('eps_phiw'),
        'ref_value_getter': lambda ref_s, s: s.get('eps_phiw',
                                                   elt='first', pos='in'),
        'post_treaters': (post_treaters.relative_difference,
                          partial(post_treaters.scale_by,
                                  scale=100., to_plot=True),
                          post_treaters.maximum),
        'tester': partial(testers.value_is_below,
                          upper_limit=20., to_plot=True),
        'markdown': r"$\Delta\epsilon_{\phi W} / \epsilon_{\phi W}$ "
                    + r"(ref $z=0$) [%]",
        'descriptor': """Longitudinal emittance should not grow by more than
                         20% along the linac.""",
        'plt_kwargs': {'fignum': 112, 'savefig': True},

    },
    "max of 99percent transverse eps_x shall not be too high": {
        'value_getter': lambda s: s.get('eps_x99'),
        'ref_value_getter': lambda ref_s, s: np.max(ref_s.get('eps_x99')),
        'post_treaters': (post_treaters.maximum,
                          partial(post_treaters.relative_difference,
                                  replace_zeros_by_nan_in_ref=False,
                                  to_plot=True)),
        'tester': partial(testers.value_is_below,
                          upper_limit=30., to_plot=True),
        'markdown': r"$\frac{max(\epsilon_{xx'}) - "
                    + r"max(\epsilon_{xx'}^{ref}))}"
                    + r"{max(\epsilon_{xx'}^{ref})}$ @ 99%",
        'descriptor': """The maximum of 99% transverse x emittance should not
                         exceed the nominal maximum of 99% transverse x
                         emittance by more than 30%.""",
        'plt_kwargs': {'fignum': 120, 'savefig': True},
    },
    "max of 99percent transverse eps_y shall not be too high": {
        'value_getter': lambda s: s.get('eps_y99'),
        'ref_value_getter': lambda ref_s, s: np.max(ref_s.get('eps_y99')),
        'post_treaters': (post_treaters.maximum,
                          partial(post_treaters.relative_difference,
                                  replace_zeros_by_nan_in_ref=False,
                                  to_plot=True)),
        'tester': partial(testers.value_is_below,
                          upper_limit=30., to_plot=True),
        'markdown': r"$\frac{max(\epsilon_{yy'}) - "
                    + r"max(\epsilon_{yy'}^{ref}))}"
                    + r"{max(\epsilon_{xx'}^{ref})}$ @ 99%",
        'descriptor': """The maximum of 99% transverse y emittance should not
                         exceed the nominal maximum of 99% transverse y
                         emittance by more than 30%.""",
        'plt_kwargs': {'fignum': 121, 'savefig': True},
    },
    "max of 99percent longitudinal eps shall not be too high": {
        'value_getter': lambda s: s.get('eps_phiw99'),
        'ref_value_getter': lambda ref_s, s: np.max(ref_s.get('eps_phiw99')),
        'post_treaters': (post_treaters.maximum,
                          partial(post_treaters.relative_difference,
                                  replace_zeros_by_nan_in_ref=False,
                                  to_plot=True)),
        'tester': partial(testers.value_is_below,
                          upper_limit=30., to_plot=True),
        'markdown': r"$\frac{max(\epsilon_{\phi W}) - "
                    + r"max(\epsilon_{\phi W}^{ref}))}"
                    + r"{max(\epsilon_{\phi W}^{ref})}$ @ 99%",
        'descriptor': """The maximum of 99% longitudinal emittance should not
                         exceed the nominal maximum of 99% longitudinal
                         emittance by more than 30%.""",
        'plt_kwargs': {'fignum': 122, 'savefig': True},

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
    "transverse eps at end": {
        'value_getter': lambda s: s.get('eps_t', elt='last', pos='out'),
        'ref_value_getter': lambda ref_s, s: ref_s.get('eps_t',
                                                       elt='last', pos='out'),
        'post_treaters': (post_treaters.relative_difference,),
        'markdown': markdown['eps_t'],
        'descriptor': """Relative difference of emittance in transverse plane
                         between fixed and reference linacs. Transverse
                         emittance defined as average of two transverse
                         planes."""
    },
    "mismatch factor at end": {
        'value_getter': lambda s: s.get('mismatch_factor',
                                        phase_space='zdelta',
                                        elt='last', pos='out'),
        'markdown': markdown['mismatch_factor'],
        'descriptor': """Mismatch factor at the end of the linac."""
    },
    "transverse mismatch factor at end": {
        'value_getter': lambda s: s.get('mismatch_factor_t',
                                        elt='last', pos='out'),
        'markdown': markdown['mismatch_factor'],
        'descriptor': """Transverse mismatch factor at the end of the linac.
                         Defined as average of two transverse mismatch
                         factors."""
    },
}


# =============================================================================
# Functions to generate presets
# =============================================================================
def presets_for_fault_scenario_rel_diff_at_some_element(
    quantity: str, elt: Element | str,
    ref_simulation_output: SimulationOutput
) -> dict[str, Callable | int | str | tuple[Callable]]:
    """
    Create the settings to evaluate a difference @ some element exit.

    Used for `FaultScenario` s.

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

    Used for `FaultScenario` s.

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
