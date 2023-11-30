#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform a test calculation with Envelope3D solver."""
import logging

import config_manager as conf_man

from core.accelerator.factory import FullStudyAcceleratorFactory
from core.accelerator.accelerator import Accelerator

from failures.fault_scenario import FaultScenario, fault_scenario_factory

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory

from visualization import plot


# =============================================================================
# Main function
# =============================================================================
if __name__ == '__main__':
    ini_filepath = '../data/from_tracewin_examples/lightwin.ini'
    ini_keys = {
        'files': 'files',
        'plots': 'plots.complete',
        'beam_calculator': 'beam_calculator.envelope_3d',
        'beam': 'beam',

    }
    my_configs = conf_man.process_config(ini_filepath, ini_keys)

    # =========================================================================
    # Beam calculators
    # =========================================================================
    beam_calculator_factory = BeamCalculatorsFactory(**my_configs)
    my_beam_calculators: tuple[BeamCalculator, ...] = \
        beam_calculator_factory.run_all()
    beam_calculators_id = beam_calculator_factory.beam_calculators_id

    # =========================================================================
    # Accelerators
    # =========================================================================
    accelerator_factory = FullStudyAcceleratorFactory(
        beam_calculators=my_beam_calculators,
        **my_configs['files'],
        **my_configs['wtf']
    )

    accelerators: list[Accelerator] = accelerator_factory.run_all()

    # =========================================================================
    # Compute propagation in nominal accelerator
    # =========================================================================
    my_beam_calculators[0].compute(accelerators[0])

    # FIXME dirty patch to initialize _element_to_index function
    if "TraceWin" in beam_calculators_id[0]:
        logging.info("Fault initialisation requires initialisation of a "
                     "sub-ListOfElements. It requires the initialisation of "
                     "a _element_to_index method, which in turn requires the "
                     "Element.beam_calc_param to be initialized. "
                     "No problem with Envelope1D, as it is performed by "
                     "Envelope1D.init_solver_parameters. "
                     "But with TraceWin, we need a first simulation to link "
                     "an index in the .out file to a position in the linac.")
        my_beam_calculators[0].compute(accelerators[1])

    # =========================================================================
    # Set up faults
    # =========================================================================
    fault_scenarios: list[FaultScenario]
    fault_scenarios = fault_scenario_factory(accelerators,
                                             my_beam_calculators[0],
                                             my_configs['wtf'],
                                             my_configs['design_space'],
                                             )

    # =========================================================================
    # Fix
    # =========================================================================
    for fault_scenario in fault_scenarios:
        fault_scenario.fix_all()

    # =========================================================================
    # Check
    # =========================================================================
    # Re-run new settings with beam_calc_pos, a priori more precise
    for accelerator in accelerators:

        ref_simulation_output = None
        if accelerator != accelerators[0] and len(my_beam_calculators) > 1:
            ref_simulation_output = \
                accelerators[0].simulation_outputs[beam_calculators_id[0]]


# =============================================================================
# Compute cumulated transfer matrix
# =============================================================================
    import numpy as np

    # Get calculated transfer matrices
    sim = accelerators[0].simulation_outputs[solv1]
    cumulated_transfer_matrices = sim.transfer_matrix.cumulated
    pos = sim.get('z_abs')

    # Get reference transfer matrices
    from tracewin_utils import load
    path = '/home/placais/LightWin/data/from_tracewin_examples/results/Transfer_matrix1.dat'
    _, ref_pos, ref_cumulated_transfer_matrices = load.transfer_matrices(path)

    # Also interpolate LightWin on TraceWin
    from util.helper import resample
    n_ref_points = ref_pos.shape[0]
    interp_cumulated = np.empty((n_ref_points, 6, 6))
    for i in range(6):
        for j in range(6):
            _, interp_cumulated[:, i, j], _, _ = resample(
                x_1=pos,
                y_1=cumulated_transfer_matrices[:, i, j],
                x_2=ref_pos,
                y_2=ref_cumulated_transfer_matrices[:, i, j])

    # Compute diff between transfer matrices
    rel_diff = 100. * (interp_cumulated - ref_cumulated_transfer_matrices) \
        / ref_cumulated_transfer_matrices
    normal_diff = interp_cumulated - ref_cumulated_transfer_matrices

    # Now a plot
    from visualization.plot import (create_fig_if_not_exists,
                                    _plot_structure,
                                    )
    i = 1
    j = 1
    fig, axes = create_fig_if_not_exists(axnum=3,
                                         sharex=True,
                                         num=25,
                                         clean_fig=True)
    axes[0].plot(ref_pos,
                 ref_cumulated_transfer_matrices[:, i, j],
                 label='Reference')
    axes[0].plot(pos,
                 cumulated_transfer_matrices[:, i, j],
                 label='Calculated')
    axes[0].legend()
    axes[0].grid(True)
    axes[0].set_ylabel(f'M {i} {j}')

    rel_plot = True
    if rel_plot:
        axes[1].plot(ref_pos, rel_diff[:, i, j])
        axes[1].set_ylabel('Rel diff %')
    else:
        axes[1].plot(ref_pos, normal_diff[:, i, j])
        axes[1].set_ylabel('Abs diff')

    axes[1].grid(True)
    _plot_structure(accelerators[0].elts, axes[2])

    tw_energy = 80.668954
    my_energy = sim.get('w_kin', elt='last', pos='out')
    print(
        f"Rel error on energy: {100. * (tw_energy - my_energy) / tw_energy}%")

    tw_beta = 0.38996284
    my_beta = sim.get('beta', elt='last', pos='out')
    print(f"Rel error on beta: {100. * (tw_beta - my_beta) / tw_beta}%")

    tw_phase = 54985.46
    my_phase = sim.get('phi_abs', elt='last', pos='out', to_deg=True)
    print(f"Rel error on phi: {100. * (tw_phase - my_phase) / tw_phase}%")
