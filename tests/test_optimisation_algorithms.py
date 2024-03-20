"""Tests the various :class:`.OptimisationAlgorithm`."""
from pathlib import Path
from typing import Any
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory

import pytest

import config_manager
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault, WithFaults

from tests.reference import compare_with_reference

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")

parameters = [
    pytest.param(('downhill_simplex', ), marks=pytest.mark.smoke),
]


@pytest.fixture(scope='module', autouse=True)
def config() -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        'files': 'files',
        'beam_calculator': 'generic_envelope1d',
        'beam': 'beam',
        'wtf': 'generic_wtf',
        'design_space': 'generic_design_space',
    }
    my_config = config_manager.process_config(config_path, config_keys)
    return my_config


def _create_solver(**config) -> BeamCalculator:
    """Instantiate the solver with the proper parameters."""
    factory = BeamCalculatorsFactory(**config)
    beam_calculator = factory.run_all()[0]
    return beam_calculator


def _create_example_accelerator(solver: BeamCalculator,
                                config: dict[str, dict[str, Any]]
                                ) -> Accelerator:
    """Create an example linac."""
    solvers = solver,
    accelerator_factory = WithFaults(beam_calculators=solvers,
                                     **config['files'],
                                     **config['wtf'])
    accelerator = accelerator_factory.run_all()[0]
    return accelerator


@pytest.fixture(scope='class', autouse=True)
def out_folder(tmp_path_factory) -> Path:
    """Create a class-scoped temporary dir for results."""
    return tmp_path_factory.mktemp('tmp')


@pytest.fixture(scope='class', params=parameters)
def simulation_output(request,
                      config: dict[str, dict[str, Any]],
                      out_folder: Path,
                      ) -> SimulationOutput:
    """Init and use a solver to propagate beam in an example accelerator."""
    optimisation_algorithm, = request.param
    config['files']['out_folder'] = out_folder
    solver = _create_solver(**config)

    accelerator = _create_example_accelerator(solver, config)
    my_simulation_output = solver.compute(accelerator)
    return my_simulation_output


@pytest.mark.implementation
class TestOptimisationAlgorithms:

    def test_something(self, simulation_output: SimulationOutput) -> None:
        """Test the initialisation."""
        assert simulation_output.has('w_kin')
