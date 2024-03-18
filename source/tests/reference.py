"""Set reference results that we should find for the reference linac."""
import numpy as np

from beam_calculation.simulation_output.simulation_output import \
    SimulationOutput

# Correspond to last values in linac, as calculated by TraceWin
REFERENCE_RESULTS = {
    'w_kin': 502.24092,
    'phi_abs': 68612.642,
    'phi_s': -24.9933,   # with new definition
    # 'phi_s': -25.0014,   # with historical definition
    'v_cav_mv': 7.85631,
    'r_xx': np.array([[+6.967351e-01, -2.406781e-01],
                      [-6.532333e-03, +2.579494e-01]]),
    'r_yy': np.array([[+6.477553e-01, -9.542593e-01],
                      [+9.573928e-02, +1.339860e-01]]),
    'r_zdelta': np.array([[+4.697375e-01, -3.121332e-01],
                          [+8.326689e-02, +3.239253e-01]]),
    'r_xy': np.array([[0., 0.], [0., 0.]]),
    'r_xz': np.array([[0., 0.], [0., 0.]]),
    'r_yz': np.array([[0., 0.], [0., 0.]]),
}


def compare_with_reference(simulation_output: SimulationOutput,
                           key: str,
                           tol: float = 1e-6,
                           **get_kwargs) -> None:
    """Compare ``key`` from ``simulation_output`` and ``REFERENCE_RESULTS``."""
    value = _get_value(simulation_output, key, **get_kwargs)
    reference_value = REFERENCE_RESULTS.get(key)

    if isinstance(value, float) and isinstance(reference_value, float):
        delta = abs(value - reference_value)
        assert delta < tol, (f"Final {key} is {value} instead of "
                             f"{reference_value} ie {delta = :.3e} > {tol = }")

    if isinstance(value, np.ndarray) and isinstance(reference_value,
                                                    np.ndarray):
        delta = np.abs(value - reference_value)
        assert np.all(delta < tol), (
            f"Final {key} is {value} instead of {reference_value}"
            f" ie {delta = } > {tol = }")


def _get_value(simulation_output: SimulationOutput,
               key: str,
               to_numpy: bool = False,
               to_deg: bool = True,
               elt: str | None = 'last',
               pos: str = 'out',
               **get_kwargs) -> float:
    """Get the data from the simulation output."""
    value = simulation_output.get(key, to_numpy=to_numpy, to_deg=to_deg,
                                  elt=elt, pos=pos, **get_kwargs)
    return value
