"""Tests the various :class:`.OptimisationAlgorithm`."""
import logging
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import WithFaults
from failures.fault_scenario import FaultScenario, fault_scenario_factory

from tests.reference import compare_with_reference

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")

parameters = [
    pytest.param(('downhill_simplex', ), marks=pytest.mark.smoke),
]


@pytest.fixture(scope='module', autouse=True)
def out_folder(tmp_path_factory) -> Path:
    """Create a class-scoped temporary dir for results."""
    return tmp_path_factory.mktemp('tmp')


@pytest.fixture(scope='module', autouse=True)
def config(out_folder: Path) -> dict[str, dict[str, Any]]:
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


@pytest.fixture(scope="module", autouse=True)
def solver(config: dict) -> BeamCalculator:
    """Instantiate the solver with the proper parameters."""
    factory = BeamCalculatorsFactory(beam_calculator=config['beam_calculator'],
                                     files=config['files'],
                                     )
    my_solver = factory.run_all()[0]
    return my_solver


def _create_accelerators(solver: BeamCalculator,
                         config: dict[str, dict[str, Any]],
                         out_folder: Path,
                         ) -> list[Accelerator]:
    """Create the reference linac and the linac that we will break."""
    solvers = solver,
    config['files']['project_folder'] = out_folder
    accelerator_factory = WithFaults(beam_calculators=solvers,
                                     **config['files'],
                                     **config['wtf'])
    accelerators = accelerator_factory.run_all()
    return accelerators


def _create_fault_scenario(accelerators: list[Accelerator],
                           solver: BeamCalculator,
                           config: dict) -> FaultScenario:
    """Create the fault(s) to fix."""
    factory = fault_scenario_factory
    fault_scenario = factory(accelerators, solver, config['wtf'],
                             config['design_space'])[0]
    return fault_scenario


@pytest.fixture(scope='class', params=parameters)
def simulation_output(request,
                      solver: BeamCalculator,
                      config: dict[str, dict[str, Any]],
                      out_folder: Path,
                      ) -> SimulationOutput:
    """Init and use a solver to propagate beam in an example accelerator."""
    optimisation_algorithm, = request.param

    accelerators = _create_accelerators(solver, config, out_folder)
    solver.compute(accelerators[0])
    fault_scenario = _create_fault_scenario(accelerators, solver, config)
    fault_scenario.fix_all()
    my_simulation_output = solver.compute(accelerators[0])
    return my_simulation_output


@pytest.mark.implementation
class TestOptimisationAlgorithms:

    def test_something(self, simulation_output: SimulationOutput) -> None:
        """Test the initialisation."""
        assert simulation_output.has('w_kin')
