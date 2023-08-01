#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 10:34:04 2023.

@author: placais
"""
import os
import pandas as pd

from core.accelerator import Accelerator


def save_files(accelerator: Accelerator,
               data_in_tracewin_style: pd.DataFrame | None = None) -> None:
    """Save a new dat with the new linac settings, as well as eval. files."""
    raise NotImplementedError
    for data, filename in zip([data_in_tracewin_style],
                              ['data_in_tracewin_style.csv']):
        if data is not None:
            out = os.path.join(accelerator.get('beam_calc_path'), filename)
            data.to_csv(out)
