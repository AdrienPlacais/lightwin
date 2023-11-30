#!/usr/bin/env python3
"""Define functions useful for beam parameters calculations."""
import numpy as np


def sigma_beam_matrices(tm_cumul: np.ndarray,
                        sigma_in: np.ndarray
                        ) -> np.ndarray:
    r"""
    Compute the :math:`\sigma` beam matrices over the linac.

    ``sigma_in`` and transfer matrices should be with the same units, in the
    same phase space.

    """
    sigma = []
    if tm_cumul.ndim == 2:
        tm_cumul = tm_cumul[np.newaxis]
    n_points = tm_cumul.shape[0]

    for i in range(n_points):
        sigma.append(tm_cumul[i] @ sigma_in @ tm_cumul[i].transpose())
    return np.array(sigma)


def reconstruct_sigma(sigma_00: np.ndarray,
                      sigma_01: np.ndarray,
                      eps: np.ndarray,
                      tol: float = 1e-8,
                      ) -> np.ndarray:
    r"""Compute sigma matrix from the two top components and emittance.

    Parameters
    ----------
    sigma_00 : np.ndarray
        Top-left component of the sigma matrix.
    sigma_01 : np.ndarray
        Top-right = bottom-left component of the sigma matrix.
    eps : np.ndarray
        Un-normalized emittance in units consistent with ``sigma_00`` and
        ``sigma_01``.
    tol : float, optional
        ``sigma_00`` is set to np.NaN where it is under ``tol`` to avoid
        ``RuntimeWarning``. The default is ``1e-8``.

    Returns
    -------
    sigma : np.ndarray
        Full sigma matrix along the linac in units consistent with
        ``sigma_00``, ``sigma_01`` and ``eps``.

    """
    sigma = np.zeros((sigma_00.shape[0], 2, 2))
    sigma_00[np.where(np.abs(sigma_00) < tol)] = np.NaN
    sigma[:, 0, 0] = sigma_00
    sigma[:, 0, 1] = sigma_01
    sigma[:, 1, 0] = sigma_01
    sigma[:, 1, 1] = (eps**2 + sigma_01**2) / sigma_00
    return sigma


def mismatch_from_arrays(ref: np.ndarray,
                         fix: np.ndarray,
                         transp: bool = False
                         ) -> np.ndarray:
    """Compute the mismatch factor between two ellipses."""
    assert isinstance(ref, np.ndarray)
    assert isinstance(fix, np.ndarray)
    # Transposition needed when computing the mismatch factor at more than one
    # position, as shape of twiss arrays is (n, 3).
    if transp:
        ref = ref.transpose()
        fix = fix.transpose()

    # R in TW doc
    __r = ref[1] * fix[2] + ref[2] * fix[1]
    __r -= 2. * ref[0] * fix[0]

    # Forbid R values lower than 2 (numerical error)
    __r = np.atleast_1d(__r)
    __r[np.where(__r < 2.)] = 2.

    mismatch = np.sqrt(.5 * (__r + np.sqrt(__r**2 - 4.))) - 1.
    return mismatch


def resample_twiss_on_fix(z_ref: np.ndarray,
                          twiss_ref: np.ndarray,
                          z_fix: np.ndarray) -> np.ndarray:
    """Interpolate ref Twiss on fix Twiss to compute mismatch afterwards."""
    n_points = z_fix.shape[0]
    out = np.empty((n_points, 3))

    for axis in range(out.shape[1]):
        out[:, axis] = np.interp(z_fix, z_ref, twiss_ref[:, axis])
    return out
