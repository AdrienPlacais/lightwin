"""Test the :class:`.TraceWin` solver.

.. todo::
    Fix :class:`.TransferMatrix` for this solver, and reintegrate the transfer
    matrix tests.

"""
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from beam_calculation.tracewin.tracewin import TraceWin
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault

from tests.reference import compare_with_reference

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")


parameters = [
    pytest.param((0, )),
    pytest.param((1, ), marks=pytest.mark.slow),
]


@pytest.fixture(scope='module', autouse=True)
def config() -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        'files': 'files',
        'beam_calculator': 'generic_tracewin',
        'beam': 'beam',
    }
    my_config = config_manager.process_config(config_path, config_keys)
    return my_config


def _create_solver(partran: int, out_folder: Path) -> TraceWin:
    """Instantiate the solver with the proper parameters."""
    base_kwargs = {"partran": partran}
    kwargs = {
        "executable": Path("/", "usr", "local", "bin", "TraceWin",
                           "./TraceWin_noX11"),
        "ini_path": DATA_DIR / "example.ini",
        "base_kwargs": base_kwargs,
        "out_folder": out_folder,
        "default_field_map_folder": DATA_DIR,
    }
    return TraceWin(**kwargs)


def _create_example_accelerator(solver: TraceWin,
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
    partran, = request.param
    solver = _create_solver(partran, out_folder)
    accelerator = _create_example_accelerator(solver, config)
    my_simulation_output = solver.compute(accelerator)
    return my_simulation_output


@pytest.mark.tracewin
class TestSolver3D:
    """Gater all the tests in a single class."""

    _w_kin_tol = 5e-3
    _phi_abs_tol = 5e0
    _phi_s_tol = 1e-6
    _v_cav_tol = 1e-7
    _r_xx_tol = 5e-1
    _r_yy_tol = 5e-1
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

    @pytest.mark.xfail(condition=True,
                       reason="TransferMatrix.get bugs w/ TraceWin",
                       raises=IndexError)
    def test_r_xx(self, simulation_output: SimulationOutput) -> None:
        """Verify that final xx transfer matrix is correct."""
        return compare_with_reference(simulation_output, 'r_xx',
                                      tol=self._r_xx_tol)

    @pytest.mark.xfail(condition=True,
                       reason="TransferMatrix.get bugs w/ TraceWin",
                       raises=IndexError)
    def test_r_yy(self, simulation_output: SimulationOutput) -> None:
        """Verify that final yy transfer matrix is correct."""
        return compare_with_reference(simulation_output, 'r_yy',
                                      tol=self._r_yy_tol)

    @pytest.mark.xfail(condition=True,
                       reason="TransferMatrix.get bugs w/ TraceWin",
                       raises=IndexError)
    def test_r_zdelta(self, simulation_output: SimulationOutput) -> None:
        """Verify that final longitudinal transfer matrix is correct."""
        return compare_with_reference(simulation_output, 'r_zdelta',
                                      tol=self._r_zdelta_tol)
