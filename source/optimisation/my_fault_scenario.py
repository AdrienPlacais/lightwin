#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:08:47 2023

@author: placais

Refactoring FaultScenario

Module holding the FaultScenario, which holds the Faults. Each Fault object
fixes himself (Fault.fix).

brok_lin: holds for "broken_linac", the linac with faults.
ref_lin: holds for "reference_linac", the ideal linac brok_lin should tend to.

TODO in neighboring_cavities, allow for preference of cavities according to
their section

TODO allow for different strategies according to the section
TODO raise explicit error when the format of error (list vs idx) is not
appropriate, especially when manual mode.

TODO tune the PSO
TODO method to avoid big changes of acceptance
TODO option to minimize the power of compensating cavities

TODO remake a small fit after the first one?
"""
import logging
import itertools
import math
import numpy as np
import pandas as pd

import config_manager as con
import optimisation.fault as mod_f
from optimisation import strategy
from core.accelerator import Accelerator
from core.list_of_elements import ListOfElements
from core.elements import FieldMap
from core.emittance import mismatch_factor
from util import debug


class MyFaultScenario(list):
    """A class to hold all fault related data."""

    def __init__(self, ref: Accelerator, fix: Accelerator,
                 wtf: dict,
                 l_fault_idx: list[int, ...] | list[list[int, ...], ...],
                 l_comp_idx: list[list[int, ...], ...] | None = None,
                 l_info_other_sol: list[dict, ...] = None) -> None:
        """
        Create the FaultScenario and the Faults.

        Parameters
        ----------
        ref : Accelerator
            Reference linac.
        fix : Accelerator
            Linac to fix.
        wtf : dict
            Holds what to fit.
        l_fault_idx : list
            List containing the position of the errors. If strategy is manual,
            it is a list of lists (faults already gathered).
        l_comp_idx : list, optional
            List containing the position of the compensating cavities. If
            strategy is manual, it must be provided. The default is None.
        l_info_other_sol : list, optional
            Contains information on another fit, for comparison purposes. The
            default is None.

        """
        self.ref, self.fix = ref, fix
        self.wtf = wtf
        self.l_info_other_sol = l_info_other_sol

        # Here, we gather faults that should be fixed together (in particular,
        # if they need the same compensating cavities)
        ll_fault_idx, ll_comp_idx = strategy.sort_and_gather_faults(
            fix, wtf, l_fault_idx, l_comp_idx)

        l_faults = []
        for l_fcav, l_ccav in zip(ll_fault_cav, ll_comp_cav):
            # linked to position
            l_elts = _zone_recomputed()
            l_faults.append(Fault(self.ref, self.fix, l_fcav, l_ccav,
                                  self.wtf))
            # Plus also l_elts, to initialize this at Fault creation

