"""Test that pickling does not raise error."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

import config_manager
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import WithFaults
from core.list_of_elements.list_of_elements import ListOfElements
from failures.fault import Fault
from failures.fault_scenario import FaultScenario, fault_scenario_factory
from util.pickling import MyCloudPickler, MyPickler

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")

params = [pytest.param((MyCloudPickler,), id="cloudpickle")]


@pytest.fixture(scope="class", params=params)
def pickler(request: pytest.FixtureRequest) -> MyPickler:
    (my_pickler_class,) = request.param
    my_pickler = my_pickler_class()
    return my_pickler


@pytest.fixture(scope="module")
def config(
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    """Set the configuration, common to all solvers."""
    out_folder = tmp_path_factory.mktemp("tmp")

    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        "files": "files",
        "beam_calculator": "generic_envelope1d",
        "beam": "beam",
        "wtf": "generic_wtf",
        "design_space": "generic_design_space",
    }
    override = {
        "files": {
            "project_folder": out_folder,
        },
    }
    my_config = config_manager.process_config(
        config_path, config_keys, warn_mismatch=True, override=override
    )
    return my_config


@pytest.fixture(scope="module")
def solver(config: dict[str, dict[str, Any]]) -> BeamCalculator:
    """Instantiate the solver with the proper parameters."""
    factory = BeamCalculatorsFactory(
        beam_calculator=config["beam_calculator"],
        files=config["files"],
    )
    my_solver = factory.run_all()[0]
    return my_solver


@pytest.fixture(scope="module")
def accelerators(
    solver: BeamCalculator,
    config: dict[str, dict[str, Any]],
) -> list[Accelerator]:
    """Create ref linac, linac we will break, compute ref simulation_output."""
    solvers = (solver,)
    accelerator_factory = WithFaults(
        beam_calculators=solvers, **config["files"], **config["wtf"]
    )
    accelerators = accelerator_factory.run_all()
    solver.compute(accelerators[0])
    return accelerators


@pytest.fixture(scope="module")
def accelerator(accelerators: Sequence[Accelerator]) -> Accelerator:
    """Return the first accelerator."""
    return accelerators[0]


@pytest.fixture(scope="module")
def list_of_elements(accelerator: Accelerator) -> ListOfElements:
    """Return a list of elements."""
    return accelerator.elts


@pytest.fixture(scope="module")
def fault_scenario(
    accelerators: list[Accelerator],
    solver: BeamCalculator,
    config: dict[str, dict[str, Any]],
) -> FaultScenario:
    """Create the fault(s) to fix."""
    factory = fault_scenario_factory
    fault_scenario = factory(
        accelerators, solver, config["wtf"], config["design_space"]
    )[0]
    return fault_scenario


@pytest.fixture(scope="module")
def fault(fault_scenario: FaultScenario) -> Fault:
    """Return the first fault of the scenario."""
    return fault_scenario[0]


# do not need to fix the error
@pytest.fixture(scope="module")
def simulation_output(
    solver: BeamCalculator,
    accelerators: list[Accelerator],
    fault_scenario: FaultScenario,
) -> SimulationOutput:
    """Get simulation output."""
    ref_simulation_output = list(accelerators[0].simulation_outputs.values())[
        0
    ]
    return ref_simulation_output


class TestMyPickler:
    """Test that pickling/unpickling does not raise errors."""

    def test_accelerator(
        self, pickler: MyPickler, accelerator: Accelerator
    ) -> None:
        """Check that :class:`.Accelerator` pickling works."""
        path = accelerator.pickle(pickler)
        pickled = Accelerator.from_pickle(pickler, path)
        assert True

    def test_list_of_elements(
        self, pickler: MyPickler, list_of_elements: ListOfElements
    ) -> None:
        """Check that :class:`.ListOfElements` pickling works."""
        path = list_of_elements.pickle(pickler)
        pickled = ListOfElements.from_pickle(pickler, path)
        assert True

    def test_fault_scenario(
        self, pickler: MyPickler, fault_scenario: FaultScenario
    ) -> None:
        """Check that :class:`.FaultScenario` pickling works."""
        path = fault_scenario.pickle(pickler)
        pickled = FaultScenario.from_pickle(pickler, path)
        assert True

    def test_fault(self, pickler: MyPickler, fault: Fault) -> None:
        """Check that :class:`.Fault` pickling works."""
        path = fault.pickle(pickler)
        pickled = Fault.from_pickle(pickler, path)
        assert True

    def test_simulation_output(
        self, pickler: MyPickler, simulation_output: SimulationOutput
    ) -> None:
        """Check that :class:`.SimulationOutput` pickling works."""
        path = simulation_output.pickle(pickler)
        pickled = SimulationOutput.from_pickle(pickler, path)
        assert True
