#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:33:39 2022.
OG lsq:
[2.02104786e-08 9.98014116e-09 9.82799129e-08] (idx 13)

@author: placais
"""
import logging
import time
import datetime

import config_manager as conf_man

from core.accelerator import Accelerator, accelerator_factory

from failures.fault_scenario import FaultScenario, fault_scenario_factory

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import create_beam_calculator_objects
from beam_calculation.output import SimulationOutput

from visualization import plot

from evaluator.list_of_simulation_output_evaluators import (
    ListOfSimulationOutputEvaluators,
    factory_simulation_output_evaluators_from_presets
)


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
    MY_CONFIG_FILE = '../data/SPIRAL2_post_scraper/lightwin.ini'
    MY_KEYS = {
        'files': 'files',
        'plots': 'plots.essential',
        'beam_calculator': 'beam_calculator.fit',
        'beam': 'beam.My_particle',
        'wtf': 'wtf.quick_debug',
        # 'beam_calculator_post': 'beam_calculator_post.tracewin',
        # 'evaluators': 'evaluators.classic',
    }
    my_configs = conf_man.process_config(MY_CONFIG_FILE, MY_KEYS)

    break_and_fix = 'wtf' in my_configs
    perform_post_simulation = 'beam_calculator_post' in my_configs
    RECOMPUTE_REFERENCE = False

    # =========================================================================
    # Set up
    # =========================================================================
    # Beam calculators
    beam_calculators_parameters = (
        my_configs['beam_calculator'],
        my_configs['beam_calculator_post'] if perform_post_simulation else None
    )
    my_beam_calculators = create_beam_calculator_objects(
        *beam_calculators_parameters)
    my_beam_calc: BeamCalculator
    my_beam_calc_post: BeamCalculator | None
    my_beam_calc, my_beam_calc_post = my_beam_calculators

    solv1 = my_beam_calc.id
    solv2 = my_beam_calc_post.id if my_beam_calc_post is not None else None

    FILEPATH = my_configs['files']['dat_file']
    PROJECT_FOLDER = my_configs['files']['project_folder']

    # Reference accelerator
    accelerators: list[Accelerator] = \
        accelerator_factory(my_beam_calculators, **my_configs)
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

    fault_scenarios: list[FaultScenario]
    fault_scenarios = fault_scenario_factory(accelerators, my_beam_calc,
                                             my_configs['wtf'])

    # =========================================================================
    # Fix
    # =========================================================================
    # for fault_scenario in fault_scenarios:
    #     start_time = time.monotonic()

    #     fault_scenario.fix_all()

    #     end_time = time.monotonic()
    #     delta_t = datetime.timedelta(seconds=end_time - start_time)
    #     logging.info(f"Elapsed time in optimisation: {delta_t}")

    # # =========================================================================
    # # Check
    # # =========================================================================
    # # Re-run new settings with beam_calc_pos, a priori more precise
    # for accelerator in accelerators:
    #     start_time = time.monotonic()

    #     ref_simulation_output = None
    #     if accelerator != accelerators[0] and solv2 is not None:
    #         ref_simulation_output = accelerators[0].simulation_outputs[solv2]
    #     post_beam_calc_and_save(accelerator, my_beam_calc_post,
    #                             ref_simulation_output=ref_simulation_output)

    #     end_time = time.monotonic()
    #     delta_t = datetime.timedelta(seconds=end_time - start_time)
    #     logging.info(f"Elapsed time in post beam calculation: {delta_t}")

    # =========================================================================
    # Post-treat
    # =========================================================================
    kwargs = {'save_fig': True, 'clean_fig': True}
    figs = plot.factory(accelerators, my_configs['plots'], **kwargs)