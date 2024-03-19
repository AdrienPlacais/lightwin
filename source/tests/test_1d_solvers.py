"""Test the 1D solvers.."""
from pathlib import Path
from typing import Any
from core.accelerator.accelerator import Accelerator

import pytest

import config_manager
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.factory import NoFault
from tests.reference import compare_with_reference

LW_DIR = Path("/home", "placais", "LightWin")
DATA_DIR = LW_DIR / "data" / "example"
TEST_DIR = LW_DIR / "source" / "tests"

testdata = [
    ('RK', False, 40),
    # ('RK', True, 40),
    # ('leapfrog', False, 60),
    # ('leapfrog', True, 60),
]


@pytest.fixture(scope='module', autouse=True)
def config() -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        'files': 'files',
        'beam_calculator': 'beam_calculator_envelope_generic',
        'beam': 'beam',
    }
    my_config = config_manager.process_config(config_path, config_keys)
    return my_config


def _create_solver(method: str,
                   flag_cython: bool,
                   n_steps_per_cell: int) -> Envelope1D:
    """Instantiate the solver with the proper parameters."""
    kwargs = {
        "flag_phi_abs": True,
        "flag_cython": flag_cython,
        "n_steps_per_cell": n_steps_per_cell,
        "method": method,
        "out_folder": TEST_DIR / "results_tests",
        "default_field_map_folder": DATA_DIR,
    }
    return Envelope1D(**kwargs)


def _create_example_accelerator(solver: Envelope1D,
                                config: dict[str, dict[str, Any]]
                                ) -> Accelerator:
    """Create an example linac."""
    accelerator_factory = NoFault(beam_calculator=solver, **config['files'])
    accelerator = accelerator_factory.run()
    return accelerator


@pytest.fixture(scope='class', params=testdata)
def simulation_output(request,
                      config: dict[str, dict[str, Any]]
                      ) -> SimulationOutput:
    """Init and use a solver to propagate beam in an example accelerator."""
    method, flag_cython, n_steps_per_cell = request.param
    solver = _create_solver(method, flag_cython, n_steps_per_cell)
    accelerator = _create_example_accelerator(solver, config)
    my_simulation_output = solver.compute(accelerator)
    return my_simulation_output


class Tests:
    """Gater all the tests in a single class."""

    def test_w_kin(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'w_kin', tol=1e-3)

    def test_phi_abs(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'phi_abs', tol=1e-2)

    def test_phi_s(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'phi_s', elt='ELT142',
                                      tol=1e-2)

    def test_v_cav(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'v_cav_mv',
                                      elt='ELT142', tol=1e-3)

    def test_r_zdelta(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'r_zdelta', tol=5e-3)
