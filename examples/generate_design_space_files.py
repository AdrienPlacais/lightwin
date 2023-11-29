#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the files used by LightWin to generate the design space.

.. todo::
    maybe show what the file should look like?

"""
from pathlib import Path
import sys
sys.path.append('/home/placais/LightWin/source')
import config_manager
import os.path

from core.accelerator.factory import StudyWithoutFaultsAcceleratorFactory
from core.elements.field_maps.field_map import FieldMap

from beam_calculation.factory import BeamCalculatorsFactory

from optimisation.design_space.factory import get_design_space_factory
from optimisation.design_space.design_space import DesignSpace


if __name__ == '__main__':
    # =========================================================================
    # Set up the accelerator
    # =========================================================================
    project_folder = Path('/home/placais/LightWin/data/MYRRHA/')
    ini_filepath = Path(project_folder, 'lightwin.ini')
    ini_keys = {
        'files': 'files',  # defines input/output files
        'beam': 'beam',    # defines beam initial properties
        # defines initial values/limits that will appear in the output files
        'design_space': 'design_space.classic',
        # define a beam calculator, in particular to compute phi_s or phi_0
        # in the nominal case
        'beam_calculator': 'beam_calculator.lightwin.envelope_longitudinal',
    }
    configuration = config_manager.process_config(ini_filepath, ini_keys)

    # =========================================================================
    # Beam calculators
    # =========================================================================
    beam_calculator_factory = BeamCalculatorsFactory(**configuration)
    beam_calculator = beam_calculator_factory.run_all()[0]
    beam_calculators_id = beam_calculator_factory.beam_calculators_id

    # =========================================================================
    # Accelerators
    # =========================================================================
    accelerator_factory = StudyWithoutFaultsAcceleratorFactory(
        beam_calculator=beam_calculator,
        **configuration['files'],
    )
    accelerator = accelerator_factory.run()

    # =========================================================================
    # Compute propagation of the beam
    # =========================================================================
    beam_calculator.compute(accelerator)

    # Take only the FieldMap objects
    cavities: list[FieldMap] = accelerator.elts.l_cav
    elements_to_put_in_file = cavities

    # =============================================================================
    # Set up generic Variable, Constraint objects
    # =============================================================================
    if configuration['design_space']['design_space_preset'] != 'everyhing':
        print("Warning! Modifying the design_space_preset entry to have all "
              "the possible variables and constraints in the output file.")
        configuration['design_space']['design_space_preset'] = 'everything'

    design_space_factory = get_design_space_factory(
        **configuration['design_space'])
    design_space = design_space_factory.run(elements_to_put_in_file,
                                            elements_to_put_in_file)

    design_space.to_files(basepath=project_folder,
                          variables_filename='variables',
                          constraints_filename='constraints',
                          overwrite=True,
                          )
    variables_filepath = os.path.join(project_folder, 'variables.csv')
    constraints_filepath = os.path.join(project_folder, 'constraints.csv')

    # Now try to create a DesignSpace from the files
    design_space = DesignSpace.from_files(('FM4', 'FM5'),
                                          variables_filepath,
                                          ('k_e', 'phi_0_abs'),
                                          None,
                                          None)
