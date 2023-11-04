#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hold the transfer matrix along the linac.

.. todo::
    Check if it can be more efficient. Maybe store R_xx, R_yy, R_zz separately?

"""
import logging
from typing import Any, Callable
import numpy as np

from core.elements.element import Element


class TransferMatrix:
    """
    Hold the (n, 6, 6) transfer matrix along the linac.

    .. note::
        When the simulation is in 1D only, the values corresponding to the
        transverse planes are filled with np.NaN.

    Attributes
    ----------
    individual : np.ndarray
        Individual transfer matrices along the linac. Not defined if not
        provided at initialisation.
    cumulated : np.ndarray
        Cumulated transfer matrices along the linac.

    """

    def __init__(self,
                 is_3d: bool,
                 first_cumulated_transfer_matrix: np.ndarray,
                 individual: np.ndarray | None = None,
                 cumulated: np.ndarray | None = None,
                 element_to_index: Callable | None = None,
                 ) -> None:
        """Create the object and compute the cumulated transfer matrix.

        Parameters
        ----------
        is_3d : bool
            If the simulation is in 3d or not.
        first_cumulated_transfer_matrix : np.ndarray
            First transfer matrix.
        individual : np.ndarray | list[np.ndarray] | None, optional
            Individual transfer matrices. The default is None, in which case
            the ``cumulated`` transfer matrix must be provided directly.
        cumulated : np.ndarray | None, optional
            Cumulated transfer matrices. The default is None, in which case the
            ``individual`` transfer matrices must be given.
        element_to_index :
            to doc

        """
        if isinstance(individual, list):
            logging.warning("should not receive a list as input")
            individual = np.array(individual)

        self.is_3d = is_3d

        self.individual: np.ndarray
        if individual is not None:
            self.individual = individual
            n_points, cumulated = self._init_from_individual(
                individual,
                first_cumulated_transfer_matrix)

        else:
            n_points, cumulated = self._init_from_cumulated(
                cumulated,
                first_cumulated_transfer_matrix)

        self.n_points = n_points

        self.cumulated = cumulated
        self._element_to_index = element_to_index

    def has(self, key: str) -> bool:
        """Check if object has attribute named ``key``."""
        return hasattr(self, key)

    def get(self,
            *keys: str,
            elt: Element | None = None,
            pos: str | None = None,
            **kwargs: Any) -> tuple[np.ndarray | float, ...]:
        """
        Shorthand to get attributes from this class or its attributes.

        Parameters
        ----------
        *keys: str
            Name of the desired attributes.
        to_numpy : bool, optional
            If you want the list output to be converted to a np.ndarray. The
            default is True.
        none_to_nan : bool, optional
            To convert None to np.NaN. The default is True.
        elt : Element | None, optional
            If provided, return the attributes only at the considered Element.
        pos : 'in' | 'out' | None
            If you want the attribute at the entry, exit, or in the whole
            Element.
        **kwargs: Any
            Other arguments passed to recursive getter.

        Returns
        -------
        out : tuple[np.ndarray | float, ...]
            Attribute(s) value(s). Will be floats if only one value is returned
            (``elt`` is given, ``pos`` is in ``('in', 'out')``).

        """
        val = {key: [] for key in keys}

        for key in keys:
            if not self.has(key):
                val[key] = None
                continue
            val[key] = getattr(self, key)

        if elt is not None:
            assert self._element_to_index is not None
            idx = self._element_to_index(elt=elt, pos=pos)
            val = {_key: _value[idx] for _key, _value in val.items()}

        if len(keys) == 1:
            return val[keys[0]]

        out = [val[key] for key in keys]
        return tuple(out)

    def _init_from_individual(
            self,
            individual: np.ndarray,
            first_cumulated_transfer_matrix: np.ndarray | None,
            ) -> tuple[int, np.ndarray]:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        individual : np.ndarray
            Individual transfer matrices along the linac.
        first_cumulated_transfer_matrix : np.ndarray | None
            First transfer matrix. It should be None if we study a linac
            from the start (``z_pos == 0.``), and should be the cumulated
            transfer matrix of the previous linac portion otherwise.

        Returns
        -------
        n_points : int
            Number of mesh points along the linac.
        cumulated : np.ndarray
            Cumulated transfer matrices.

        """
        n_points = individual.shape[0] + 1
        if self.is_3d:
            shape = (n_points, 6, 6)
        else:
            shape = (n_points, 2, 2)

        if first_cumulated_transfer_matrix is None:
            first_cumulated_transfer_matrix = np.eye(shape[1])

        cumulated = self._compute_cumulated(first_cumulated_transfer_matrix,
                                            shape,
                                            self.is_3d,
                                            n_points)
        return n_points, cumulated

    def _init_from_cumulated(
            self,
            cumulated: np.ndarray | None,
            first_cumulated_transfer_matrix: np.ndarray,
            tol: float = 1e-8
            ) -> tuple[int, np.ndarray]:
        """Check that the given cumulated matrix is valid.

        Parameters
        ----------
        cumulated : np.ndarray
            Cumulated transfer matrices along the linac.
        first_cumulated_transfer_matrix : np.ndarray
            The first of the cumulated transfer matrices.
        tol : float, optional
            The max allowed difference between ``cumulated`` and
            ``first_cumulated_transfer_matrix`` when determining if they are
            the same or not.

        Returns
        -------
        n_points : int
            Number of mesh points along the linac.
        cumulated : np.ndarray
            Cumulated transfer matrices.

        """
        if cumulated is None:
            logging.error("You must provide at least one of the two "
                          "arrays: individual transfer matrices or "
                          "cumulated transfer matrices.")
            raise IOError("Wrong input")
        n_points = cumulated.shape[0]

        if (np.abs(cumulated[0]
                   - first_cumulated_transfer_matrix)).any() > tol:
            n_points += 1
            cumulated = np.vstack((first_cumulated_transfer_matrix[np.newaxis],
                                   cumulated))

        return n_points, cumulated

    def _compute_cumulated(self,
                           first_cumulated_transfer_matrix: np.ndarray,
                           shape: tuple[int, int, int],
                           is_3d: bool,
                           n_points: int) -> np.ndarray:
        """Compute cumulated transfer matrix from individual.

        Parameters
        ----------
        first_cumulated_transfer_matrix : np.ndarray
            First transfer matrix. It should be eye matrix if we study a linac
            from the start (``z_pos == 0.``), and should be the cumulated
            transfer matrix of the previous linac portion otherwise.
        shape : tuple[int, int, int]
            Shape of the output ``cumulated`` array.
        is_3d : bool
            If the simulation is in 3D or not.
        n_points : int
            Number of mesh points along the linac.

        Returns
        -------
        cumulated : np.ndarray
            Cumulated transfer matrix.

        .. todo::
            I think the 3D/1D handling may be smarter?

        """
        cumulated = np.full(shape, np.NaN)
        cumulated[0] = first_cumulated_transfer_matrix

        for i in range(n_points - 1):
            cumulated[i + 1] = self.individual[i] @ cumulated[i]

        if is_3d:
            return cumulated

        cumulated_1d = cumulated
        cumulated = np.full((n_points, 6, 6), np.NaN)
        cumulated[:, 4:, 4:] = cumulated_1d
        return cumulated

    @property
    def r_xx(self) -> np.ndarray:
        """Return the transfer matrix of :math:`[x-x']` plane."""
        return self.cumulated[:, :2, :2]

    @r_xx.setter
    def r_xx(self, r_xx: np.ndarray) -> None:
        """Set the transfer matrix of :math:`[x-x']` plane."""
        self.cumulated[:, :2, :2] = r_xx

    @property
    def r_yy(self) -> np.ndarray:
        """Return the transfer matrix of :math:`[y-y']` plane."""
        return self.cumulated[:, 2:4, 2:4]

    @r_yy.setter
    def r_yy(self, r_yy: np.ndarray) -> None:
        """Set the transfer matrix of :math:`[y-y']` plane."""
        self.cumulated[:, 2:4, 2:4] = r_yy

    @property
    def r_zz(self) -> np.ndarray:
        r"""Return the transfer matrix of :math:`[z-\delta]` plane.

        .. deprecated:: v3.2.2.3
            Use ``r_zdelta`` instead. Although it is called ``r_zz`` in the
            TraceWin doc, it is a transfer matrix in the :math:`[z-\delta]`
            plane.

        """
        return self.cumulated[:, 4:, 4:]

    @r_zz.setter
    def r_zz(self, r_zz: np.ndarray) -> None:
        r"""Set the transfer matrix of :math:`[z-\delta]` plane.

        .. deprecated:: v3.2.2.3
            Use ``r_zdelta`` instead. Although it is called ``r_zz`` in the
            TraceWin doc, it is a transfer matrix in the :math:`[z-\delta]`
            plane.

        """
        self.cumulated[:, 4:, 4:] = r_zz

    @property
    def r_zdelta(self) -> np.ndarray:
        r"""Return the transfer matrix of :math:`[z-\delta]` plane."""
        return self.cumulated[:, 4:, 4:]

    @r_zdelta.setter
    def r_zdelta(self, r_zdelta: np.ndarray) -> None:
        r"""Set the transfer matrix of :math:`[z-\delta]` plane."""
        self.cumulated[:, 4:, 4:] = r_zdelta
