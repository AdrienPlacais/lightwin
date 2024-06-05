"""Test that some commands do what they are supposed to do."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import config_manager
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import NoFault

params = [
    pytest.param(
        ("repeat_ele.dat", "generic_envelope3d"),
        marks=pytest.mark.implementation,
        id="test REPEAT_ELE command",
    ),
]

DATA_DIR = Path("data", "test_instructions")
TEST_DIR = Path("tests")


@pytest.fixture(scope="class", params=params)
def config(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    out_folder = tmp_path_factory.mktemp("tmp")
    dat_file, beam_calculator_key = request.param

    config_path = DATA_DIR / "test_instructions.toml"
    config_keys = {
        "files": "files",
        "beam_calculator": beam_calculator_key,
        "beam": "beam",
    }
    override = {
        "files": {
            "project_folder": out_folder,
            "dat_file": dat_file,
        },
    }
    my_config = config_manager.process_config(
        config_path, config_keys, warn_mismatch=True, override=override
    )
    return my_config


@pytest.fixture
def solver(config: dict[str, dict[str, Any]]) -> BeamCalculator:
    """Instantiate the solver with the proper parameters."""
    factory = BeamCalculatorsFactory(
        beam_calculator=config["beam_calculator"],
        files=config["files"],
    )
    my_solver = factory.run_all()[0]
    return my_solver


@pytest.fixture
def accelerator(
    solver: BeamCalculator,
    config: dict[str, dict[str, Any]],
) -> Accelerator:
    """Create an example linac."""
    accelerator_factory = NoFault(beam_calculator=solver, **config["files"])
    accelerator = accelerator_factory.run()
    return accelerator


@pytest.fixture
def simulation_output(
    solver: BeamCalculator,
    accelerator: Accelerator,
) -> SimulationOutput:
    """Init and use a solver to propagate beam in an example accelerator."""
    my_simulation_output = solver.compute(accelerator)
    return my_simulation_output


class TestCommands:
    """Just test that REPEAT_ELE leads to proper transfer matrix.

    .. todo::
        A whole class for a single test? There are better options... Should
        only have as arguments: params and expected transfer matrix

    """

    def test_repeat_element(self, simulation_output: SimulationOutput) -> None:
        """Verify that final transfer matrix is correct."""
        expected = np.array(
            [
                [
                    +4.810796e-01,
                    +8.860652e-02,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                ],
                [
                    -8.673880e00,
                    +4.810796e-01,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                ],
                [
                    +0.000000e00,
                    +0.000000e00,
                    +1.626705e00,
                    +1.341772e-01,
                    +0.000000e00,
                    +0.000000e00,
                ],
                [
                    +0.000000e00,
                    +0.000000e00,
                    +1.226863e01,
                    +1.626705e00,
                    +0.000000e00,
                    +0.000000e00,
                ],
                [
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +1.000000e00,
                    +1.097659e-01,
                ],
                [
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +0.000000e00,
                    +1.000000e00,
                ],
            ]
        )
        transfer_matrix = simulation_output.transfer_matrix
        assert transfer_matrix is not None
        returned = transfer_matrix.cumulated
        assert np.allclose(
            expected, returned, atol=1e-10
        ), f"{expected = }, while {returned = }"
