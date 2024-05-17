"""Define several helper functions.

.. todo::
    Should they be in a class somewhere?

"""
from collections.abc import Collection
from typing import Any

from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.simulation_output.simulation_output import SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault


def compute_beam(
    beam_calculator: BeamCalculator,
    config_files: dict[str, Any],
) -> tuple[Accelerator, SimulationOutput]:
    """Create the :class:`.Accelerator` and compute beam in it.

    Parameters
    ----------
    beam_calculator : BeamCalculator
        Solver to use.
    config_files : dict[str, Any]
        Holds info on file input/output.

    Returns
    -------
    Accelerator
        An accelerator with its :class:`.SimulationOutput`.

    """
    accelerator_factory = NoFault(
        beam_calculator=beam_calculator, **config_files
    )

    accelerator = accelerator_factory.run()
    simulation_output = beam_calculator.compute(accelerator)
    return accelerator, simulation_output


def compute_beams(
    beam_calculators: Collection[BeamCalculator], config_files: dict[str, Any]
) -> tuple[list[Accelerator], list[SimulationOutput]]:
    """Propagate beam with all :class:`.BeamCalculator`."""
    accelerators, simulation_outputs = zip(
        *(
            compute_beam(beam_calculator, config_files)
            for beam_calculator in beam_calculators
        )
    )
    return accelerators, simulation_outputs
