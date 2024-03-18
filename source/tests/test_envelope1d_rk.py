"""Test the Cython and not-Cython versions of :class:`.Envelope1D`."""
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import StudyWithoutFaultsAcceleratorFactory
from tests.reference import compare_with_reference


@pytest.fixture(scope="module")
def config() -> dict[str, dict[str, Any]]:
    """Load and prepare the standard configuration. Created once for module."""
    config_path = Path("..", "..", "data", "example", "lightwin.toml")
    config_keys = {
        'files': 'files',
        'beam_calculator': 'beam_calculator_envelope_generic',
        'beam': 'beam',
    }
    my_config = config_manager.process_config(config_path, config_keys)
    return my_config


@pytest.fixture(scope="module")
def solver() -> Envelope1D:
    """Create a standard RK solver. Created once for the module."""
    kwargs = {
        "flag_phi_abs": True,
        "flag_cython": False,
        "n_steps_per_cell": 40,
        "method": 'RK',
        "out_folder": Path('results_tests/'),
        "default_field_map_folder": Path('..', '..', 'data', 'example'),
    }
    return Envelope1D(**kwargs)


@pytest.fixture(scope="module")
def accelerator(solver: Envelope1D,
                config: dict[str, dict[str, Any]]) -> Accelerator:
    """Create a standard accelerator."""
    accelerator_factory = StudyWithoutFaultsAcceleratorFactory(
        beam_calculator=solver,
        **config['files']
    )
    accelerator = accelerator_factory.run()
    return accelerator


@pytest.fixture(scope="module")
def simulation_output(solver: Envelope1D, accelerator: Accelerator
                      ) -> SimulationOutput:
    """Compute propagation in the example linac."""
    simulation_output = solver.compute(accelerator)
    return simulation_output


def test_is_1d(solver: Envelope1D) -> None:
    """Test if the solver is in 1D."""
    assert not solver.is_a_3d_simulation


def test_is_envelope(solver: Envelope1D) -> None:
    """Check that the simulation is not multiparticle."""
    assert not solver.is_a_multiparticle_simulation


def test_w_kin(simulation_output: SimulationOutput) -> None:
    """Verify that final energy is correct."""
    return compare_with_reference(simulation_output, 'w_kin', tol=1e-3)


def test_phi_abs(simulation_output: SimulationOutput) -> None:
    """Verify that final energy is correct."""
    return compare_with_reference(simulation_output, 'phi_abs', tol=1e-2)


def test_phi_s(simulation_output: SimulationOutput) -> None:
    """Verify that final energy is correct."""
    return compare_with_reference(simulation_output, 'phi_s', elt='ELT142',
                                  tol=1e-2)


def test_v_cav(simulation_output: SimulationOutput) -> None:
    """Verify that final energy is correct."""
    return compare_with_reference(simulation_output, 'v_cav_mv', elt='ELT142',
                                  tol=1e-3)


def test_r_zdelta(simulation_output: SimulationOutput) -> None:
    """Verify that final energy is correct."""
    return compare_with_reference(simulation_output, 'r_zdelta', tol=5e-3)
