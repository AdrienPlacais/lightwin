"""Test the :class:`.Envelope1D` solver."""
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault

from tests.reference import compare_with_reference

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")

leapfrog_marker = pytest.mark.xfail(
    condition=True,
    reason="leapfrog method has not been updated since 0.0.0.0.01 or so")

parameters = [
    pytest.param(('RK', False, 40), marks=pytest.mark.smoke),
    pytest.param(('RK', True, 40), marks=pytest.mark.cython),
    pytest.param(('leapfrog', False, 60), marks=leapfrog_marker),
    pytest.param(('leapfrog', True, 60), marks=(leapfrog_marker,
                                                pytest.mark.cython)),
]


@pytest.fixture(scope='module', autouse=True)
def config() -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        'files': 'files',
        'beam_calculator': 'generic_envelope1d',
        'beam': 'beam',
    }
    my_config = config_manager.process_config(config_path, config_keys)
    return my_config


def _create_solver(method: str,
                   flag_cython: bool,
                   n_steps_per_cell: int,
                   out_folder: Path) -> Envelope1D:
    """Instantiate the solver with the proper parameters."""
    kwargs = {
        "flag_phi_abs": True,
        "flag_cython": flag_cython,
        "n_steps_per_cell": n_steps_per_cell,
        "method": method,
        "out_folder": out_folder,
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
    method, flag_cython, n_steps_per_cell = request.param
    solver = _create_solver(method, flag_cython, n_steps_per_cell, out_folder)
    accelerator = _create_example_accelerator(solver, config)
    my_simulation_output = solver.compute(accelerator)
    return my_simulation_output


class TestSolver1D:
    """Gater all the tests in a single class."""

    _w_kin_tol = 1e-3
    _phi_abs_tol = 1e-2
    _phi_s_tol = 1e-2
    _v_cav_tol = 1e-3
    _r_zdelta_tol = 5e-3

    def test_w_kin(self, simulation_output: SimulationOutput) -> None:
        """Verify that final energy is correct."""
        return compare_with_reference(simulation_output, 'w_kin',
                                      tol=self._w_kin_tol)

    def test_phi_abs(self, simulation_output: SimulationOutput) -> None:
        """Verify that final absolute phase is correct."""
        return compare_with_reference(simulation_output, 'phi_abs',
                                      tol=self._phi_abs_tol)

    def test_phi_s(self, simulation_output: SimulationOutput) -> None:
        """Verify that synchronous phase in last cavity is correct."""
        return compare_with_reference(simulation_output, 'phi_s', elt='ELT142',
                                      tol=self._phi_s_tol)

    def test_v_cav(self, simulation_output: SimulationOutput) -> None:
        """Verify that accelerating voltage in last cavity is correct."""
        return compare_with_reference(simulation_output, 'v_cav_mv',
                                      elt='ELT142', tol=self._v_cav_tol)

    def test_r_zdelta(self, simulation_output: SimulationOutput) -> None:
        """Verify that final longitudinal transfer matrix is correct."""
        return compare_with_reference(simulation_output, 'r_zdelta',
                                      tol=self._r_zdelta_tol)
