#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:08:47 2023.

@author: placais

Refactoring FaultScenario
"""
import logging
import numpy as np

from optimisation.my_fault import MyFault
from optimisation import strategy, position
from core.accelerator import Accelerator
from core.list_of_elements import ListOfElements
from core.emittance import mismatch_factor


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
        logging.warning('still a transfer phase to implement')
        self._transfer_phi0_from_ref_to_broken()

    def fix_all(self) -> None:
        """Fix all the Faults."""
        success, info = [], []
        for fault in self:
            _succ, _info = fault.fix()
            success.append(_succ)
            info.append(_info)

            my_sol = _info['X']
            self._compute_beam_parameters_in_compensation_zone_and_save_it(fault, my_sol)

            results, elts = \
                self._compute_beam_parameters_up_to_next_fault(fault, my_sol)
            self.fix_acc.store_results(results, elts)


        results = self.fix_acc.elts.compute_transfer_matrices()
        results['mismatch factor'] = self._compute_mismatch()
        self.fix_acc.store_results(results, self.fix_acc.elts)
        self.fix_acc.name = f"Fixed ({str(success.count(True))}" \
            + f" of {str(len(success))})"

        logging.warning("Removed some debugs here.")
        # for linac in [self.ref_acc, self.fix_Acc]:
            # self.info[linac.name + ' cav'] = \
                # debug.output_cavities(linac, mod_f.debugs['cav'])

        # self.info['fit'] = debug.output_fit(self, mod_f.debugs['fit_complete'],
                                            # mod_f.debugs['fit_compact'])

    def _transfer_phi0_from_ref_to_broken(self) -> None:
        """
        Transfer the entry phases from ref linac to broken.

        If the absolute initial phases are not kept between reference and
        broken linac, it comes down to rephasing the linac. This is what we
        want to avoid when con.FLAG_PHI_ABS = True.
        """
        # Get CavitieS of REFerence and BROKen linacs
        ref_cavities = self.ref_acc.l_cav#.get_elts('nature', 'FIELD_MAP')
        fix_cavities = self.fix_acc.l_cav#get_elts('nature', 'FIELD_MAP')

        for ref_cavity, fix_cavity in zip(ref_cavities, fix_cavities):
            ref_a_f = ref_cavity.acc_field
            fix_a_f = fix_cavity.acc_field

            fix_a_f.phi_0['phi_0_abs'] = ref_a_f.phi_0['phi_0_abs']
            fix_a_f.phi_0['phi_0_rel'] = ref_a_f.phi_0['phi_0_rel']
            fix_a_f.phi_0['nominal_rel'] = ref_a_f.phi_0['phi_0_rel']

    def _compute_beam_parameters_in_compensation_zone_and_save_it(self, fault: MyFault, d_fits: dict) -> None:
        results = fault.elts.compute_transfer_matrices(d_fits=d_fits, transfer_data=True)
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
                  'phi_s fit': True}
        logging.warning("Here again, phi_s_fit not handled.")

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
