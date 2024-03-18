"""Test the Cython and not-Cython versions of :class:`.Envelope1D`."""
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.envelope_1d.envelope_1d import Envelope1D
from core.accelerator.factory import StudyWithoutFaultsAcceleratorFactory
from core.list_of_elements.list_of_elements import ListOfElements


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
        "out_folder": Path('..', 'test-results'),
        "default_field_map_folder": Path(
            '..', '..', 'data', 'example', 'field_maps_1D'),
        "flag_phi_abs": True,
        "flag_cython": False,
        "n_steps_per_cell": 40,
        "method": 'RK',
    }
    return Envelope1D(**kwargs)


@pytest.fixture
def elements(solver: Envelope1D,
             config: dict[str, dict[str, Any]]) -> ListOfElements:
    """Create a standard list of elements."""
    accelerator_factory = StudyWithoutFaultsAcceleratorFactory(
        beam_calculator=solver,
        **config['files']
    )
    accelerator = accelerator_factory.run()
    elements = accelerator.elts
    return elements


def test_is_1d(solver: Envelope1D) -> None:
    """Test if the solver is in 1D."""
    assert not solver.is_a_3d_simulation


def test_is_envelope(solver: Envelope1D) -> None:
    """Check that the simulation is not multiparticle."""
    assert not solver.is_a_multiparticle_simulation


def test_elts(elements: ListOfElements) -> None:
    """Check how the list of elements is considered."""
    assert elements[0].name == 'QP1'
