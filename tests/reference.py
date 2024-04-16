"""Set reference results that we should find for the reference linac."""

import numpy as np

from beam_calculation.simulation_output.simulation_output import (
    SimulationOutput,
)

# Correspond to last values in linac, as calculated by TraceWin
REFERENCE_RESULTS = {
    "w_kin": 502.24092,
    "phi_abs": 68510.456,
    "phi_s": -24.9933,  # with new definition
    # 'phi_s': -25.0014,   # with historical definition
    "v_cav_mv": 7.85631,
    "r_xx": np.array(
        [
            [
                +1.214036e00,
                -2.723429e00,
            ],
            [
                +2.090116e-01,
                -3.221306e-01,
            ],
        ]
    ),
    "r_yy": np.array(
        [[-1.453483e-01, -1.022289e00], [+1.503132e-01, -1.684692e-01]]
    ),
    "r_zdelta": np.array(
        [[+4.509904e-01, -3.843910e-01], [+9.079210e-02, +3.176355e-01]]
    ),
    "r_xy": np.array([[0.0, 0.0], [0.0, 0.0]]),
    "r_xz": np.array([[0.0, 0.0], [0.0, 0.0]]),
    "r_yz": np.array([[0.0, 0.0], [0.0, 0.0]]),
}


def compare_with_reference(
    simulation_output: SimulationOutput,
    key: str,
    tol: float = 1e-6,
    **get_kwargs,
) -> None:
    """Compare ``key`` from ``simulation_output`` and ``REFERENCE_RESULTS``."""
    value = _get_value(simulation_output, key, **get_kwargs)
    reference_value = REFERENCE_RESULTS.get(key)

    if isinstance(value, float) and isinstance(reference_value, float):
        delta = abs(value - reference_value)
        assert delta < tol, (
            f"Final {key} is {value:.3e} instead of "
            f"{reference_value:.3e} ie {delta = :.3e} > "
            f"{tol = :.1e}"
        )

    if isinstance(value, np.ndarray) and isinstance(
        reference_value, np.ndarray
    ):
        delta = np.abs(value - reference_value)
        assert np.all(delta < tol), (
            f"Final {key} is {value} instead of {reference_value}"
            f" ie {delta = } > {tol = }"
        )


def compare_with_other(
    ref_so: SimulationOutput,
    fix_so: SimulationOutput,
    key: str,
    tol: float = 1e-6,
    **get_kwargs,
) -> None:
    """Compare ``key`` from ``simulation_output`` and ``REFERENCE_RESULTS``."""
    value = _get_value(fix_so, key, **get_kwargs)
    reference_value = _get_value(ref_so, key, **get_kwargs)

    if isinstance(value, float) and isinstance(reference_value, float):
        delta = abs(value - reference_value)
        assert delta < tol, (
            f"Final {key} is {value:.3e} instead of "
            f"{reference_value:.3e} ie {delta = :.3e} > "
            f"{tol = :.1e}"
        )

    if isinstance(value, np.ndarray) and isinstance(
        reference_value, np.ndarray
    ):
        delta = np.abs(value - reference_value)
        assert np.all(delta < tol), (
            f"Final {key} is {value} instead of {reference_value}"
            f" ie {delta = } > {tol = }"
        )


def _get_value(
    simulation_output: SimulationOutput,
    key: str,
    to_numpy: bool = False,
    to_deg: bool = True,
    elt: str | None = "last",
    pos: str = "out",
    **get_kwargs,
) -> float:
    """Get the data from the simulation output."""
    value = simulation_output.get(
        key, to_numpy=to_numpy, to_deg=to_deg, elt=elt, pos=pos, **get_kwargs
    )
    return value
