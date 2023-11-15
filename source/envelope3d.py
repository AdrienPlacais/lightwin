#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Perform a test calculation with Envelope3D solver."""
import logging
import time
import datetime

import config_manager as conf_man

from core.accelerator import Accelerator, accelerator_factory

from failures.fault_scenario import FaultScenario, fault_scenario_factory

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import create_beam_calculator_objects
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

from visualization import plot


def _wrap_beam_calculation(accelerator: Accelerator,
                           beam_calculator: BeamCalculator,
                           **kwargs: SimulationOutput
                           ) -> SimulationOutput:
    """Shorthand to init the solver, perform beam calculation."""
    beam_calculator.init_solver_parameters(accelerator)
    simulation_output = beam_calculator.run(accelerator.elts)
    simulation_output.compute_complementary_data(accelerator.elts, **kwargs)
    return simulation_output


def beam_calc_and_save(accelerator: Accelerator,
                       beam_calculator: BeamCalculator,
                       **kwargs: SimulationOutput):
    """Perform the simulation, save it into Accelerator.simulation_output."""
    simulation_output = _wrap_beam_calculation(accelerator, beam_calculator,
                                               **kwargs)
    accelerator.keep_settings(simulation_output)
    accelerator.keep_simulation_output(simulation_output, beam_calculator.id)


def post_beam_calc_and_save(accelerator: Accelerator,
                            beam_calculator: BeamCalculator | None,
                            recompute_reference: bool = True,
                            **kwargs: SimulationOutput):
    """Perform the simulation, save it into Accelerator.simulation_output."""
    if beam_calculator is None:
        return

    beam_calculator.init_solver_parameters(accelerator)
    if accelerator.name == 'Working' and not recompute_reference:
        logging.info("Not recomputing reference linac. Implement the auto "
                     "taking from a reference folder.")
        return

    simulation_output = _wrap_beam_calculation(accelerator, beam_calculator)
    accelerator.keep_simulation_output(simulation_output, beam_calculator.id)


# =============================================================================
# Main function
# =============================================================================
if __name__ == '__main__':
    MY_CONFIG_FILE = '../data/from_tracewin_examples/lightwin.ini'
    MY_KEYS = {
        'files': 'files',
        'plots': 'plots.complete',
        'beam_calculator': 'beam_calculator.envelope_3d',
        'beam': 'beam',
    }
    my_configs = conf_man.process_config(MY_CONFIG_FILE, MY_KEYS)

    break_and_fix = 'wtf' in my_configs
    RECOMPUTE_REFERENCE = False

    # =========================================================================
    # Set up
    # =========================================================================
    # Beam calculators
    beam_calculators_parameters = (my_configs['beam_calculator'], )
    my_beam_calculators = create_beam_calculator_objects(
        *beam_calculators_parameters)
    my_beam_calc: BeamCalculator = my_beam_calculators[0]
    solv1 = my_beam_calc.id

    FILEPATH = my_configs['files']['dat_file']
    PROJECT_FOLDER = my_configs['files']['project_folder']

    # Reference accelerator
    accelerators: list[Accelerator] = accelerator_factory(my_beam_calculators,
                                                          **my_configs)
    beam_calc_and_save(accelerators[0], my_beam_calc)
    # FIXME dirty patch to initialize _element_to_index function
    if "TraceWin" in solv1:
        logging.info("Fault initialisation requires initialisation of a "
                     "sub-ListOfElements. It requires the initialisation of "
                     "a _element_to_index method, which in turn requires the "
                     "Element.beam_calc_param to be initialized. "
                     "No problem with Envelope1D, as it is performed by "
                     "Envelope1D.init_solver_parameters. "
                     "But with TraceWin, we need a first simulation to link "
                     "an index in the .out file to a position in the linac.")
        beam_calc_and_save(accelerators[1], my_beam_calc)

    # =========================================================================
    # Post-treat
    # =========================================================================
    kwargs = {'save_fig': True, 'clean_fig': True}
    figs = plot.factory([accelerators[0], accelerators[0]],
                        my_configs['plots'],
                        **kwargs)


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
    from visualization.plot import (_create_fig_if_not_exists,
                                    _plot_structure,
                                    )
    i = 1
    j = 1
    fig, axes = _create_fig_if_not_exists(axnum=3,
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
    _plot_structure(accelerators[0], axes[2])

    tw_energy = 80.668954
    my_energy = sim.get('w_kin', elt='last', pos='out')
    print(f"Rel error on energy: {100. * (tw_energy - my_energy) / tw_energy}%")

    tw_beta = 0.38996284
    my_beta = sim.get('beta', elt='last', pos='out')
    print(f"Rel error on beta: {100. * (tw_beta - my_beta) / tw_beta}%")

    tw_phase = 54985.46
    my_phase = sim.get('phi_abs', elt='last', pos='out', to_deg=True)
    print(f"Rel error on phi: {100. * (tw_phase - my_phase) / tw_phase}%")
