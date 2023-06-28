#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 14:33:39 2022.

@author: placais
"""

# import os
import logging
# from copy import deepcopy
import time
import datetime
# import pandas as pd

import config_manager as conf_man
from core.accelerator import Accelerator, accelerator_factory
from optimisation.fault_scenario import FaultScenario, fault_scenario_factory
# import tracewin.interface
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import create_beam_calculator_object
from beam_calculation.output import SimulationOutput
# from util import evaluate
from visualization import plot


def _wrap_beam_calculation(accelerator: Accelerator,
                           beam_calculator: BeamCalculator
                           ) -> SimulationOutput:
    """Shorthand to init the solver, perform beam calculation."""
    beam_calculator.init_solver_parameters(accelerator)
    elts = accelerator.elts
    simulation_output = beam_calculator.run(elts)
    simulation_output.compute_complementary_data(elts)
    return simulation_output


def beam_calc_and_save(accelerator: Accelerator,
                       beam_calculator: BeamCalculator):
    """Perform the simulation, save it into Accelerator.simulation_output."""
    simulation_output = _wrap_beam_calculation(accelerator, beam_calculator)
    accelerator.keep_settings(simulation_output)
    accelerator.simulation_output = simulation_output


def post_beam_calc_and_save(accelerator: Accelerator,
                            beam_calculator: BeamCalculator | None,
                            recompute_reference: bool = True):
    """Perform the simulation, save it into Accelerator.simulation_output."""
    if beam_calculator is None:
        return

    beam_calculator.init_solver_parameters(accelerator)
    if accelerator.name == 'Working' and not recompute_reference:
        logging.info("Not recomputing reference linac. Implement the auto "
                     "taking from a reference folder.")
        return

    # simulation_output = _wrap_beam_calculation(accelerator, beam_calculator)

    # elts = accelerator.elts
    logging.warning("Need to give TraceWin.run_with_this and run a dat "
                    "filepath. Inconsistent with Envelope1D!!")
    simulation_output = beam_calculator.run(accelerator.elts,
                                            accelerator.get('dat_filepath'))
    # simulation_output.compute_complementary_data(elts)

    accelerator.simulation_output_post = simulation_output

    # lin.files["out_tw"] = os.path.join(os.path.dirname(FILEPATH),
    #                                    'ref')
    # logging.info(
    #     "we do not TW recompute reference accelerator. "
    #     + f"We take TW results from {lin.files['out_tw']}.")
    # continue
    # output.save_files(accelerator,
    #                   data_in_tracewin_style=simulation_output.in_tw_fashion)


# =============================================================================
# Main function
# =============================================================================
if __name__ == '__main__':
    MY_CONFIG_FILE = 'myrrha.ini'
    MY_KEYS = {
        'files': 'files',
        'plots': 'plots.essential',
        'beam_calculator': 'beam_calculator.lightwin.envelope_longitudinal',
        'beam': 'beam',
        'wtf': 'wtf.k_out_of_n',
        # 'beam_calculator_post': 'beam_calculator_post.tracewin.quick_debug',
    }
    my_configs = conf_man.process_config(MY_CONFIG_FILE, MY_KEYS)

    break_and_fix = 'wtf' in my_configs
    perform_post_simulation = 'beam_calculator_post' in my_configs
    RECOMPUTE_REFERENCE = False

    # =========================================================================
    # Set up BeamCalculator objects
    # =========================================================================
    my_beam_calc: BeamCalculator
    my_beam_calc = create_beam_calculator_object(my_configs['beam_calculator'])

    my_beam_calc_post: BeamCalculator | None
    my_beam_calc_post = create_beam_calculator_object(
        my_configs['beam_calculator_post']) \
        if perform_post_simulation else None

    FILEPATH = my_configs['files']['dat_file']
    PROJECT_FOLDER = my_configs['files']['project_folder']

    accelerators: list[Accelerator] = accelerator_factory(**my_configs)
    beam_calc_and_save(accelerators[0], my_beam_calc)

    fault_scenarios: list[FaultScenario]
    fault_scenarios = fault_scenario_factory(accelerators, my_beam_calc,
                                             my_configs['wtf'])
    for fault_scenario in fault_scenarios:
        start_time = time.monotonic()

        fault_scenario.fix_all()

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time in optimisation: {delta_t}")

    for accelerator in accelerators:
        start_time = time.monotonic()

        post_beam_calc_and_save(accelerator, my_beam_calc_post)

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time in post beam calculation: {delta_t}")

        # ini_path = FILEPATH.replace('.dat', '.ini')
        # TODO transfer ini path elsewhere
        # tw_simu = TraceWinBeamCalculator(post_tw['executable'],
        #                                  ini_path,
        #                                  lin.get('out_tw'),
        #                                  lin.get('dat_filepath'),
        #                                  post_tw)

#     if 'Fixed' in lin.name:
#         tracewin.interface.resample_tracewin_results(
#             ref=accelerators[0].tracewin_simulation,
#             fix=lin.tracewin_simulation)

#     if 'Fixed' in lin.name:
#         d_fred = evaluate.fred_tests(accelerators[0], lin)
#         l_fred.append(d_fred)

#         d_bruce = evaluate.bruce_tests(accelerators[0], lin)
#         l_bruce.append(d_bruce)

# if break_and_fix:
#     for _list, name in zip([l_fred, l_bruce],
#                            ['fred_tests.csv', 'bruce_tests.csv']):
#         out = pd.DataFrame(_list)
#         filepath = os.path.join(PROJECT_FOLDER, name)
#         out.to_csv(filepath)

# =============================================================================
# Plot
# =============================================================================
    kwargs = {'plot_tw': perform_post_simulation, 'save_fig': False,
              'clean_fig': True}
    for i in range(len(fault_scenarios)):
        for str_plot, to_plot in my_configs['plots'].items():
            if not to_plot:
                continue
            # Plot the reference accelerator, i-th broken accelerator and
            # corresponding fixed accelerator
            if not break_and_fix:
                args = (accelerators[0], )
            else:
                # args = (accelerators[0], accelerators[2 * i + 1],
                # accelerators[2 * i + 2])
                args = (accelerators[0], accelerators[i + 1])
            plot.plot_preset(str_plot, *args, **kwargs)
