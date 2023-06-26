#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:08:47 2023.

@author: placais

This module holds FaultScenario, a list-based class holding all the Fault
objets to be fixed.
"""
import logging
from typing import Any
import os.path

import numpy as np
import pandas as pd

import config_manager as con
from beam_calculation.beam_calculator import BeamCalculator
from optimisation.fault import Fault
from optimisation import strategy, position
from core.elements import _Element
from core.accelerator import Accelerator
from util import debug, helper

DISPLAY_CAVITIES_INFO = True


class FaultScenario(list):
    """A class to hold all fault related data."""

    def __init__(self, ref_acc: Accelerator, fix_acc: Accelerator,
                 beam_calculator: BeamCalculator,
                 wtf: dict, fault_idx: list[int] | list[list[int]],
                 comp_idx: list[list[int]] | None = None,
                 info_other_sol: list[dict] = None) -> None:
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
        fault_idx : list
            List containing the position of the errors. If strategy is manual,
            it is a list of lists (faults already gathered).
        comp_idx : list, optional
            List containing the position of the compensating cavities. If
            strategy is manual, it must be provided. The default is None.
        info_other_sol : list, optional
            Contains information on another fit, for comparison purposes. The
            default is None.

        """
        self.ref_acc, self.fix_acc = ref_acc, fix_acc
        self.beam_calculator = beam_calculator
        self.wtf = wtf
        self.info_other_sol = info_other_sol
        self.info = {}

        gathered_fault_idx, gathered_comp_idx = \
            strategy.sort_and_gather_faults(fix_acc, wtf, fault_idx, comp_idx)

        faults = []
        for fault, comp in zip(gathered_fault_idx, gathered_comp_idx):
            elts_subset, objectives_positions = \
                position.compensation_zone(fix_acc, wtf, fault, comp)

            faulty_cavities = [fix_acc.l_cav[i] for i in fault]
            compensating_cavities = [fix_acc.l_cav[i] for i in comp]

            faults.append(
                Fault(self.ref_acc, self.fix_acc, self.wtf, faulty_cavities,
                      compensating_cavities, elts_subset,
                      objectives_positions)
            )
        super().__init__(faults)

        if not con.FLAG_PHI_ABS:
            # Change status of cavities after the first one that is down. Idea
            # is to keep relative phi_0 between ref and fix linacs (linac
            # rephasing)
            self._update_status_of_cavities_to_rephase()

        self._transfer_phi0_from_ref_to_broken()

    def fix_all(self) -> None:
        """Fix all the Faults."""
        success, info = [], []
        ref_twiss_zdelta = self.ref_acc.get('twiss_zdelta')
        for fault in self:
            fault.update_cavities_status(optimisation='not started')
            _succ, optimized_cavity_settings, _info = fault.fix(
                self.beam_calculator.run_with_this)

            success.append(_succ)
            info.append(_info)

            # Now we recompute full linac
            simulation_output = self.beam_calculator.run_with_this(
                optimized_cavity_settings, self.fix_acc.elts)
            self.fix_acc.keep_this(simulation_output,
                                   ref_twiss_zdelta=ref_twiss_zdelta)
            fault.get_x_sol_in_real_phase()
            fault.update_cavities_status(optimisation='finished', success=True)

            if not con.FLAG_PHI_ABS:
                # Tell LW to keep the new phase of the rephased cavities
                # between the two compensation zones
                self._reupdate_status_of_rephased_cavities(fault)

        self.fix_acc.name = f"Fixed ({str(success.count(True))}" \
            + f" of {str(len(success))})"

        for linac in [self.ref_acc, self.fix_acc]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, DISPLAY_CAVITIES_INFO)

        self._evaluate_fit_quality(save=True)

        # Legacy, does not work anymore with the new implementation
        # self.info['fit'] = debug.output_fit(self, FIT_COMPLETE, FIT_COMPACT)

    def _update_status_of_cavities_to_rephase(self) -> None:
        """
        Change the status of some cavities to 'rephased'.

        If the calculation is in relative phase, all cavities that are after
        the first failed one are rephased.
        """
        logging.warning(
            "The phases in the broken linac are relative. It may be more "
            + "relatable to use absolute phases, as it would avoid the "
            + "rephasing of the linac at each cavity.")
        cavities = self.fix_acc.l_cav
        first_failed_cavity = self[0].failed_cav[0]
        first_failed_index = cavities.index(first_failed_cavity)

        cavities_to_rephase = [cav for cav in cavities[first_failed_index:]
                               if cav.get('status') == 'nominal']

        for cav in cavities_to_rephase:
            cav.update_status('rephased (in progress)')

    def _reupdate_status_of_rephased_cavities(self, fault: Fault) -> None:
        """
        Modify the status of the cavities that were already rephased.

        Change the cavities with status "rephased (in progress)" to
        "rephased (ok)" between the fault in argument and the next one.
        """
        logging.warning("Changed the way of defining idx1 and idx2.")
        elts = self.fix_acc.elts

        idx1 = fault.elts[-1].idx['elt_idx']
        idx2 = len(elts)
        if fault is not self[-1]:
            next_fault = self[self.index(fault) + 1]
            idx2 = next_fault.elts[0].idx['elt_idx'] + 1

        rephased_cavities_between_two_faults = [
            elt for elt in elts[idx1:idx2]
            if elt.get('nature') == 'FIELD_MAP'
            and elt.get('status') == 'rephased (in progress)']

        for cav in rephased_cavities_between_two_faults:
            cav.update_status('rephased (ok)')

    def _transfer_phi0_from_ref_to_broken(self) -> None:
        """
        Transfer the entry phases from ref linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when con.FLAG_PHI_ABS = True.
        """
        ref_cavities = self.ref_acc.l_cav
        fix_cavities = self.fix_acc.l_cav

        for ref_cavity, fix_cavity in zip(ref_cavities, fix_cavities):
            ref_a_f = ref_cavity.acc_field
            fix_a_f = fix_cavity.acc_field

            fix_a_f.phi_0['phi_0_abs'] = ref_a_f.phi_0['phi_0_abs']
            fix_a_f.phi_0['phi_0_rel'] = ref_a_f.phi_0['phi_0_rel']
            fix_a_f.phi_0['nominal_rel'] = ref_a_f.phi_0['phi_0_rel']

    # FIXME could be simpler
    def _evaluate_fit_quality(self, save: bool = True,
                              additional_elt: list[_Element] | None = None
                              ) -> None:
        """
        Compute some quantities on the whole linac to see if fit is good.

        Parameters
        ----------
        save : bool, optional
            To tell if you want to save the evaluation. The default is True.
        additional_elt : list[_Element] | None, optional
            If you want to evaluate the quality of the beam at the exit of
            additional _Elements. The default is None.

        """
        quantities_to_evaluate = [
            'w_kin', 'phi_abs', 'envelope_pos_w', 'envelope_energy_w',
            'mismatch_factor', 'eps_w']
        quantities = {key: [] for key in quantities_to_evaluate}

        evaluation_elt = [fault.elts[-1] for fault in self]
        headers = [f"end comp zone\n({elt = }) [%]" for elt in evaluation_elt]

        if additional_elt is not None:
            headers += [f"user defined\n({elt = }) [%]"
                        for elt in additional_elt]
            evaluation_elt += additional_elt

        headers.append("end linac [%]")
        evaluation_elt.append(self.fix_acc.elts[-1])

        headers.insert(0, "Qty")

        # Calculate relative errors in %
        for elt in evaluation_elt:
            for key in quantities_to_evaluate:
                fix = self.fix_acc.get(key, elt=elt, pos='out')

                if key == 'mismatch_factor':
                    quantities[key].append(fix)
                    continue

                ref = self.ref_acc.get(key, elt=elt, pos='out')
                quantities[key].append(1e2 * (ref - fix) / ref)

        headers.append("sum error linac")
        for key in quantities_to_evaluate:
            fix = self.fix_acc.get(key)

            if key == 'mismatch_factor':
                quantities[key].append(np.sum(fix))
                continue

            ref = self.ref_acc.get(key)
            ref[ref == 0.] = np.NaN

            quantities[key].append(np.nansum(np.sqrt(((ref - fix) / ref)**2)))

        # Now make it a pandas dataframe for sweet output
        df_eval = pd.DataFrame(columns=headers)
        for i, key in enumerate(quantities_to_evaluate):
            df_eval.loc[i] = [key] + quantities[key]
        logging.info(helper.pd_output(df_eval, header='Fit evaluation'))

        if save:
            out = os.path.join(self.fix_acc.get('out_lw'),
                               'settings_quality_tests.csv')
            df_eval.to_csv(out)


def fault_scenario_factory(accelerators: list[Accelerator],
                           beam_calculator: BeamCalculator,
                           wtf: dict[str, Any]) -> list[FaultScenario]:
    """Shorthand to generate the FaultScenario objects."""
    scenarios_fault_idx = wtf.pop('failed')

    scenarios_comp_idx = [None for accelerator in accelerators[1:]]
    if 'manual list' in wtf:
        scenarios_comp_idx = wtf.pop('manual list')

    _ = [beam_calculator._init_solver_parameters(accelerator.elts)
         for accelerator in accelerators]

    fault_scenarios = [FaultScenario(ref_acc=accelerators[0],
                                     fix_acc=accelerator,
                                     beam_calculator=beam_calculator,
                                     wtf=wtf,
                                     fault_idx=fault_idx,
                                     comp_idx=comp_idx)
                       for accelerator, fault_idx, comp_idx
                       in zip(accelerators[1:], scenarios_fault_idx,
                              scenarios_comp_idx)]

    return fault_scenarios
