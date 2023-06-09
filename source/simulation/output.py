#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 11:56:20 2023.

@author: placais

A class to uniformly store the outputs from the different simulation tools:
    envelope1d
    tracewin_envelope
    tracewin_multiparticle
"""
from dataclasses import dataclass

import numpy as np


# TODO remove unnecessary
@dataclass
class SimulationOutput:
    """Stores the information that is needed for a fit."""

    w_kin: list[float] | None = None
    phi_abs_array: list[float] | None = None
    mismatch_factor: list[float | None] | None = None

    cavity_parameters: list[dict | None] | None = None
    phi_s: list[float] | None = None
    individual_transfer_matrices: list[np.ndarray(2)] | None = None
    cumulated_transfer_matrices: np.ndarray(3) | None = None
    rf_fields: list[dict] | None = None
    eps_zdelta: np.ndarray(1) | None = None
    twiss_zdelta: np.ndarray(2) | None = None
    sigma_matrix: np.ndarray(3) | None = None
