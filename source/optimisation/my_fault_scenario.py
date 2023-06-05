#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:08:47 2023.

@author: placais

Refactoring FaultScenario
"""
import logging
import numpy as np
import pandas as pd

import config_manager as con
from optimisation.my_fault import MyFault
from optimisation import strategy, position
from core.accelerator import Accelerator
from core.list_of_elements import ListOfElements
from core.emittance import mismatch_factor
from util import debug

DISPLAY_CAVITIES_INFO = True


class MyFaultScenario(list):
    """A class to hold all fault related data."""

    def __init__(self, ref_acc: Accelerator, fix_acc: Accelerator,
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
        self.wtf = wtf
        self.info_other_sol = info_other_sol
        self.info = {}

        gathered_fault_idx, gathered_comp_idx = \
            strategy.sort_and_gather_faults(fix_acc, wtf, fault_idx, comp_idx)

        faults = []
        for fault, comp in zip(gathered_fault_idx, gathered_comp_idx):
            elts_subset, objectives_positions = \
                position.compensation_zone(fix_acc, wtf, fault, comp)
            # Here objectives_positions is Element index
            # Ultimately I'll need solver index (envelope) or Element index
            # (TW)
            # WARNING! mesh index will be different from ref to fix... Maybe it
            # would be better to stick to the exit of an _Element name
            faulty_cavities = [fix_acc.l_cav[i] for i in fault]
            compensating_cavities = [fix_acc.l_cav[i] for i in comp]
            faults.append(
                MyFault(self.ref_acc, self.fix_acc, self.wtf, faulty_cavities,
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
        for fault in self:
            fault.update_cavities_status(optimisation='not started')
            _succ, _info = fault.fix()
            success.append(_succ)
            info.append(_info)

            my_sol = _info['X']
            self._compute_beam_parameters_in_compensation_zone_and_save_it(
                fault, my_sol)

            fault.update_cavities_status(optimisation='finished', success=True)
            results, elts = \
                self._compute_beam_parameters_up_to_next_fault(fault, my_sol)
            self.fix_acc.store_results(results, elts)

            logging.error("removed some phi transfer here")

        results = self.fix_acc.elts.compute_transfer_matrices()
        results['mismatch factor'] = self._compute_mismatch()
        self.fix_acc.store_results(results, self.fix_acc.elts)
        self.fix_acc.name = f"Fixed ({str(success.count(True))}" \
            + f" of {str(len(success))})"

        for linac in [self.ref_acc, self.fix_acc]:
            self.info[linac.name + ' cav'] = \
                debug.output_cavities(linac, DISPLAY_CAVITIES_INFO)

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

    def _compute_beam_parameters_in_compensation_zone_and_save_it(
            self, fault: MyFault, sol: list) -> None:
        d_fits = {'l_phi': sol[:len(sol) // 2],
                  'l_k_e': sol[len(sol) // 2:],
                  'phi_s fit': self.wtf['phi_s fit']}
        results = fault.elts.compute_transfer_matrices(d_fits=d_fits,
                                                       transfer_data=True)
        self.fix_acc.store_results(results, fault.elts)
        fault.get_x_sol_in_real_phase()

    def _compute_beam_parameters_up_to_next_fault(
            self, fault: MyFault, my_sol: dict) -> tuple[dict, ListOfElements]:
        """Compute propagation up to last element of the next fault."""
        first_elt = fault.elts[-1]
        last_elt = self.fix_acc.elts[-1]
        if fault is not self[-1]:
            idx = self.index(fault)
            last_elt = self[idx + 1].elts[-1]

        __elts = self.fix_acc.elts[
            first_elt.get('elt_idx', to_numpy=False):
            last_elt.get('elt_idx', to_numpy=False) + 1]
        idx_in = first_elt.get('s_in', to_numpy=False)
        w_kin = self.fix_acc.get('w_kin')[idx_in]
        phi_abs = self.fix_acc.get('phi_abs_array')[idx_in]
        transf_mat = self.fix_acc.get('tm_cumul')[idx_in]

        elts = ListOfElements(__elts, w_kin, phi_abs, idx_in, transf_mat)

        # FIXME
        d_fits = {'l_phi': my_sol[:len(my_sol) // 2],
                  'l_k_e': my_sol[len(my_sol) // 2:],
                  'phi_s fit': self.wtf['phi_s fit']}

        results = elts.compute_transfer_matrices(d_fits=d_fits,
                                                 transfer_data=True)
        return results, elts

    def _compute_mismatch(self) -> np.ndarray:
        """
        Compute the mismatch between reference abnd broken linac.

        Also store it into the broken_linac.beam_param dictionary.
        """
        mism = mismatch_factor(self.ref_acc.get("twiss_z"),
                               self.fix_acc.get("twiss_z"), transp=True)
        return mism

    # FIXME
    def evaluate_fit_quality(self, delta_t: float, user_idx: int = None
                             ) -> pd.DataFrame:
        """Compute some quantities on the whole linac to see if fit is good."""
        keys = ['w_kin', 'phi_abs_array', 'envelope_pos_w',
                'envelope_energy_w', 'mismatch factor', 'eps_w']
        val = {}
        for key in keys:
            val[key] = []

        # End of each compensation zone
        l_idx = [fault.elts[-1].get('s_out') for fault in self]
        str_columns = [f"end comp zone\n(idx {idx}) [%]"
                       for idx in l_idx]

        # If user provided more idx to check
        if user_idx is not None:
            l_idx += user_idx
            str_columns += [f"user defined\n(idx {idx}) [%]"
                            for idx in user_idx]

        # End of linac
        l_idx.append(-1)
        str_columns.append("end linac [%]")

        # First column labels
        str_columns.insert(0, "Qty")

        # Calculate relative errors in %
        for idx in l_idx:
            for key in keys:
                ref = self.ref_acc.get(key)[idx]
                fix = self.fix_acc.get(key)[idx]

                if key == 'mismatch factor':
                    val[key].append(fix)
                    continue
                val[key].append(1e2 * (ref - fix) / ref)

        # Relative square difference sumed on whole linac
        str_columns.append("sum error linac")
        for key in keys:
            ref = self.ref_acc.get(key)
            ref[ref == 0.] = np.NaN
            fix = self.fix_acc.get(key)

            if key == 'mismatch factor':
                val[key].append(np.sum(fix))
                continue
            val[key].append(np.nansum(np.sqrt(((ref - fix) / ref)**2)))

        # Handle time
        time_line = [None for n in range(len(val[keys[0]]))]
        days, seconds = delta_t.days, delta_t.seconds
        time_line[0] = f"{days * 24 + seconds // 3600} hrs"
        time_line[1] = f"{seconds % 3600 // 60} min"
        time_line[2] = f"{seconds % 60} sec"

        # Now make it a pandas dataframe for sweet output
        df_eval = pd.DataFrame(columns=str_columns)
        df_eval.loc[0] = ['time'] + time_line
        for i, key in enumerate(keys):
            df_eval.loc[i + 1] = [key] + val[key]

        return df_eval
