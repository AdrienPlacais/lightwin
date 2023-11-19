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
import config_manager

from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import FullStudyAcceleratorFactory

from failures.fault_scenario import FaultScenario, fault_scenario_factory

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory

from visualization import plot

from evaluator.list_of_simulation_output_evaluators import (
    ListOfSimulationOutputEvaluators,
    factory_simulation_output_evaluators_from_presets
)


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
        'design_space': 'design_space.quick_debug',
        # 'wtf': 'wtf.k_out_of_n',
        # 'beam_calculator_post': 'beam_calculator_post.tracewin.quick_debug',
        # 'evaluators': 'evaluators.fred',
    }
    my_configs = config_manager.process_config(ini_filepath, ini_keys)

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
