"""Define Runge-Kutta integraiton function."""

from collections.abc import Callable

import numpy as np


def rk4(
    u: np.ndarray,
    du: Callable[[float, np.ndarray], np.ndarray],
    x: float,
    dx: float,
) -> np.ndarray:
    """Compute variation of ``u`` between ``x`` and ``x+dx``.

    Use 4-th order Runge-Kutta method.

    Note
    ----
    This is a slightly modified version of the RK. The ``k_i`` are proportional
    to ``delta_u`` instead of ``du_dz``.

    Parameters
    ----------
    u : np.ndarray
        Holds the value of the function to integrate in ``x``.
    du_dx : Callable[[float, np.ndarray], np.ndarray]
        Gives the variation of ``u`` components with ``x``.
    x : float
        Where ``u`` is known.
    dx : float
        Integration step.

    Return
    ------
    delta_u : np.ndarray
        Variation of ``u`` between ``x`` and ``x+dx``.

    """
    half_dx = 0.5 * dx
    k_1 = du(x, u)
    k_2 = du(x + half_dx, u + 0.5 * k_1)
    k_3 = du(x + half_dx, u + 0.5 * k_2)
    k_4 = du(x + dx, u + k_3)
    delta_u = (k_1 + 2.0 * k_2 + 2.0 * k_3 + k_4) / 6.0
    return delta_u
