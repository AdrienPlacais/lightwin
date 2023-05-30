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
from optimisation.my_fault import MyFault
from optimisation import strategy, position
from core.accelerator import Accelerator


class MyFaultScenario(list):
    """A class to hold all fault related data."""

    def __init__(self, ref_acc: Accelerator, fix_acc: Accelerator,
                 wtf: dict, l_fault_idx: list[int] | list[list[int]],
                 l_comp_idx: list[list[int]] | None = None,
                 l_info_other_sol: list[dict] = None) -> None:
        """
        Create the FaultScenario and the Faults.

        Parameters
        ----------
        ref_acc : Accelerator
            Reference linac.
        fix_acc : Accelerator
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
        self.ref_acc, self.fix_acc = ref_acc, fix_acc
        self.wtf = wtf
        self.l_info_other_sol = l_info_other_sol

        # Here, we gather faults that should be fixed together (in particular,
        # if they need the same compensating cavities)
        ll_fault_idx, ll_comp_idx = strategy.sort_and_gather_faults(
            fix_acc, wtf, l_fault_idx, l_comp_idx)

        l_faults = []
        for l_fidx, l_cidx in zip(ll_fault_idx, ll_comp_idx):
            l_elts, l_check = position.compensation_zone(fix_acc, wtf, l_fidx,
                                                         l_cidx)
            # Here l_check is Element index
            # Ultimately I'll need solver index (envelope) or Element index
            # (TW)
            # WARNING! mesh index will be different from ref to fix... Maybe it
            # would be better to stick to the exit of an _Element name
            l_fcav = [fix_acc.elts[i] for i in l_fidx]
            l_ccav = [fix_acc.elts[i] for i in l_cidx]
            l_faults.append(
                MyFault(self.ref_acc, self.fix_acc, self.wtf, l_fcav, l_ccav,
                        l_elts, l_check)
            )
            # warning: sometimes, a compensating cavity is in the compensation
            # zone, but is is dedicated to another fault

