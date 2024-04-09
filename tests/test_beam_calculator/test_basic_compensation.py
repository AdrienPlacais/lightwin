"""Test that all :class:`.BeamCalculator` can be used for compensation."""

from pathlib import Path
from typing import Any

import config_manager
import pytest
from beam_calculation.beam_calculator import BeamCalculator
from beam_calculation.factory import BeamCalculatorsFactory
from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)
from core.accelerator.accelerator import Accelerator
from core.accelerator.factory import WithFaults
from failures.fault_scenario import FaultScenario, fault_scenario_factory

from tests.reference import compare_with_other

DATA_DIR = Path("data", "example")
TEST_DIR = Path("tests")

params = [
    pytest.param(("generic_envelope1d",), marks=pytest.mark.smoke),
    pytest.param(("generic_envelope3d",), marks=pytest.mark.smoke),
    pytest.param(("generic_tracewin",), marks=pytest.mark.smoke),
]


@pytest.fixture(scope="class", params=params)
def config(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, dict[str, Any]]:
    """Set the configuration."""
    out_folder = tmp_path_factory.mktemp("tmp")
    (solver_key,) = request.param

    config_path = DATA_DIR / "lightwin.toml"
    config_keys = {
        "files": "files",
        "beam_calculator": solver_key,
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
        config_path,
        config_keys,
        warn_mismatch=True,
        override=override,
    )
    return my_config


@pytest.fixture(scope="class")
def solver(config: dict[str, dict[str, Any]]) -> BeamCalculator:
    """Instantiate the solver with the proper parameters."""
    factory = BeamCalculatorsFactory(
        beam_calculator=config["beam_calculator"],
        files=config["files"],
    )
    my_solver = factory.run_all()[0]
    return my_solver


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
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


@pytest.fixture(scope="class")
def simulation_outputs(
    solver: BeamCalculator,
    accelerators: list[Accelerator],
    fault_scenario: FaultScenario,
) -> tuple[SimulationOutput, SimulationOutput]:
    """Get ref simulation output, fix fault, compute fix simulation output."""
    ref_simulation_output = list(accelerators[0].simulation_outputs.values())[0]
    fault_scenario.fix_all()
    fix_simulation_output = solver.compute(accelerators[1])
    return ref_simulation_output, fix_simulation_output


@pytest.mark.implementation
class TestAllBeamCalculatorCanCompensate:

    _w_kin_tol = 1e-3
    _phi_abs_tol = 1e-2
    _phi_s_tol = 1e-2
    _v_cav_tol = 1e-3
    _r_zdelta_tol = 5e-3

    def test_w_kin(
        self, simulation_outputs: tuple[SimulationOutput, SimulationOutput]
    ) -> None:
        """Test the initialisation."""
        return compare_with_other(
            *simulation_outputs, key="w_kin", tol=self._w_kin_tol
        )

    def test_phi_abs(
        self, simulation_outputs: tuple[SimulationOutput, SimulationOutput]
    ) -> None:
        """Test the initialisation."""
        return compare_with_other(
            *simulation_outputs, key="phi_abs", tol=self._phi_abs_tol
        )

    def test_phi_s(
        self, simulation_outputs: tuple[SimulationOutput, SimulationOutput]
    ) -> None:
        """Test the initialisation."""
        return compare_with_other(
            *simulation_outputs, key="phi_s", tol=self._phi_s_tol, elt="ELT142"
        )

    def test_v_cav(
        self, simulation_outputs: tuple[SimulationOutput, SimulationOutput]
    ) -> None:
        """Test the initialisation."""
        return compare_with_other(
            *simulation_outputs,
            key="v_cav_mv",
            tol=self._v_cav_tol,
            elt="ELT142",
        )
