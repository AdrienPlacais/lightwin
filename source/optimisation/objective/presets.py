#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:17:35 2023.

@author: placais

Here we define the presets to easily initialize the :class:`Objective` objects.

"""

from optimisation.objective.minimize_difference_with_ref import \
    MinimizeDifferenceWithRef
from optimisation.objective.mismatch import Mismatch

from util.dicts_output import d_markdown


# =============================================================================
# Classic settings for ADS
# =============================================================================
ads_1 = [
    MinimizeDifferenceWithRef,
    {
        'name': d_markdown['w_kin'],
        'weight': 1.,
        'get_key': 'w_kin',
        'get_kwargs': {'elt': 11111111111, 'pos': 'out', 'to_numpy': False},
        'descriptor': """Minimize diff. of w_kin between ref and fix at the end
        of the compensation zone.
        """
    }
]

ads_2 = [
    MinimizeDifferenceWithRef,
    {
        'name': d_markdown['phi_abs'].replace('deg', 'rad'),
        'weight': 1.,
        'get_key': 'phi_abs',
        'get_kwargs': {'elt': 11111111111, 'pos': 'out', 'to_numpy': False},
        'descriptor': """Minimize diff. of phi_abs between ref and fix at the
        end of the compensation zone.
        """
    }
]

ads_3 = [
    Mismatch,
    {
        'name': 'Mismatch in [z-delta] plane',
        'weight': 1.,
        'get_key': 'twiss',
        'get_kwargs': {'elt': 1111111,
                       'pos': 'out',
                       'to_numpy': True,
                       'phase_space': 'zdelta'},
        'descriptor': """Minimize mismatch factor in [z-delta] plane."""
    }

]

ads_settings = [ads_1, ads_2, ads_3]
