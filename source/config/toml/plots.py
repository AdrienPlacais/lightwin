#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Define the functions to test that plots will work.

The presets listed here are defined in :mod:`visualization.plot`. This is where
you will find corresponding documentation, and where you should define you own
presets.


.. todo::
    It would be interesting to be able to pass some fig_kw directly from the
    ``.toml``. I guess it is one of the reason of why I started refactoring the
    plots...

"""
import logging

from config.toml.helper import check_type

IMPLEMENTED_PLOTS = ('energy', 'phase', 'cav', 'emittance', 'twiss',
                     'envelopes', 'transfer matrices')  #:
TO_UPDATE_OR_FIX = ('transfer matrices', )  #:


def test(energy: bool = False,
         phase: bool = False,
         cav: bool = False,
         emittance: bool = False,
         twiss: bool = False,
         envelopes: bool = False,
         transfer_matrices: bool = False,
         **plots_kw: bool) -> None:
    """Check if the desired plots are implemented."""
    check_type(bool, 'plots', energy, phase, cav, emittance, twiss, envelopes,
               transfer_matrices)
    if len(plots_kw) > 0:
        logging.warning(f"Some plots were not recognized: {plots_kw = }."
                        "They probably will not do anything.")


def add_some_values(plots_kw: dict[str, str]) -> None:
    """Just pass."""
    pass
