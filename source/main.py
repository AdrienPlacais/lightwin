#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the holds a generic compensation workflow.

.. todo::
    phi_s fit is included in design_space preset

.. todo::
    Proper example file, maybe with a Jupyter Notebook.

.. todo::
    Too many responsibilities in this script!!

"""
import logging
import time
import datetime

import config_manager

from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import FullStudyAcceleratorFactory

from failures.fault_scenario import FaultScenario, fault_scenario_factory

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

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
                            beam_calculator: BeamCalculator,
                            recompute_reference: bool = True,
                            **kwargs: SimulationOutput):
    """Perform the simulation, save it into Accelerator.simulation_output."""

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
    ini_filepath = '../data/MYRRHA/lightwin.ini'
    ini_keys = {
        'files': 'files',
        'plots': 'plots.complete',
        'beam_calculator': 'beam_calculator.lightwin.envelope_longitudinal',
        # 'beam_calculator': 'beam_calculator.tracewin.envelope',
        'beam': 'beam',
        'wtf': 'wtf.quick_debug',
        # 'wtf': 'wtf.k_out_of_n',
        # 'beam_calculator_post': 'beam_calculator_post.tracewin.quick_debug',
        # 'evaluators': 'evaluators.fred',
    }
    my_configs = config_manager.process_config(ini_filepath, ini_keys)

    RECOMPUTE_REFERENCE = False

    # =========================================================================
    # Beam calculators
    # =========================================================================
    beam_calculator_factory = BeamCalculatorsFactory(**my_configs)
    my_beam_calculators = beam_calculator_factory.run_all()
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
    beam_calc_and_save(accelerators[0], my_beam_calculators[0])
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
        beam_calc_and_save(accelerators[1], my_beam_calculators[0])

    # =========================================================================
    # Set up faults
    # =========================================================================
    fault_scenarios: list[FaultScenario]
    fault_scenarios = fault_scenario_factory(accelerators,
                                             my_beam_calculators[0],
                                             my_configs['wtf'])

    # =========================================================================
    # Fix
    # =========================================================================
    for fault_scenario in fault_scenarios:
        start_time = time.monotonic()
        fault_scenario.fix_all()
        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time in optimisation: {delta_t}")

    # =========================================================================
    # Check
    # =========================================================================
    # Re-run new settings with beam_calc_pos, a priori more precise
    for accelerator in accelerators:
        start_time = time.monotonic()

        ref_simulation_output = None
        if accelerator != accelerators[0] and len(my_beam_calculators) > 1:
            ref_simulation_output = \
                accelerators[0].simulation_outputs[beam_calculators_id[1]]
            post_beam_calc_and_save(accelerator, my_beam_calculators[1],
                                    ref_simulation_output=ref_simulation_output)

        end_time = time.monotonic()
        delta_t = datetime.timedelta(seconds=end_time - start_time)
        logging.info(f"Elapsed time in post beam calculation: {delta_t}")

    # =========================================================================
    # Post-treat
    # =========================================================================
    kwargs = {'save_fig': True, 'clean_fig': True}
    figs = plot.factory(accelerators, my_configs['plots'], **kwargs)

    # s_to_study = [accelerator.simulation_outputs[solv2]
    #               for accelerator in accelerators]
    # ref_s = s_to_study[0]

    # simulation_output_evaluators: ListOfSimulationOutputEvaluators = \
    #     factory_simulation_output_evaluators_from_presets(
    #         *my_configs['evaluators']['beam_calc_post'],
    #         ref_simulation_output=ref_s)

    # simulation_output_evaluators.run(*tuple(s_to_study))
