"""Define an utility function to compare two :class:`.BeamCalculator`.

.. todo::
    Allow for undetermined number of BeamCalculator in the config, and update
    here.

"""

from collections.abc import Collection
from pathlib import Path
from typing import Any

import config_manager
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault
from core.elements.element import Element
from visualization import plot


def output_comparison(
    sim_1: SimulationOutput,
    sim_2: SimulationOutput,
    element: Element,
    qty: str,
    single_value: bool,
    **kwargs,
) -> str:
    """Compare two simulation outputs.

    Parameters
    ----------
    sim1, sim2 : SimulationOutput
        Objects to compate.
    element : Element
        Element at which look for ``qty``.
    qty : str
        Quantity that will be compared.
    single_value : bool
        True if a single value is expected, False if it is an array.

    Returns
    -------
    msg : str
        Holds requested information.

    """
    kwargs = {"to_deg": True, "elt": element}

    if single_value:
        msg = f"""
        Comparing {qty} in {element}.
        1: {sim_1.get(qty, **kwargs)}
        2: {sim_2.get(qty, **kwargs)}
        """
        return msg

    msg = f"""
    Comparing {qty} in {element}.
    LW: {sim_1.get(qty, **kwargs)[0]} to {sim_1.get(qty, **kwargs)[-1]}
    TW: {sim_2.get(qty, **kwargs)[0]} to {sim_2.get(qty, **kwargs)[-1]}
    """
    return msg


def _compute_beam(
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
            _compute_beam(beam_calculator, config_files)
            for beam_calculator in beam_calculators
        )
    )
    return accelerators, simulation_outputs


def main(
    toml_filepath: Path,
    toml_keys: dict[str, str],
    tests: Collection[dict[str, str | bool]],
) -> None:
    """Compute beam with two beam calculators and compare them."""
    configuration = config_manager.process_config(toml_filepath, toml_keys)

    beam_calculator_factory = BeamCalculatorsFactory(**configuration)
    beam_calculators = beam_calculator_factory.run_all()

    accelerators, simulation_outputs = compute_beams(
        beam_calculators, configuration["files"]
    )

    kwargs = {"save_fig": True, "clean_fig": True}
    _ = plot.factory(accelerators, configuration["plots"], **kwargs)

    for test in tests:
        output_comparison(simulation_outputs, **test)


if __name__ == "__main__":
    toml_filepath = Path("lightwin.toml")
    toml_keys = {
        "files": "files",
        "plots": "plots_complete",
        "beam_calculator": "envelope1d",
        "beam_calculator_post": "tracewin_envelope",
        "beam": "beam",
    }
    tests = (
        {"element": "DR50", "qty": "phi_abs", "single_value": False},
        {"element": "DR50", "qty": "w_kin", "single_value": False},
        {"element": "DR50", "qty": "phi_s", "single_value": True},
    )
    main(toml_filepath, toml_keys, tests)
